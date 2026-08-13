"""
Organic Dataset Construction Script
====================================
Pulls organic (non-LLM-generated) content from five public sources — MS MARCO v2.1,
FeTaQA, CoSQA, Enron Email, and CNN/DailyMail — and builds two datasets:

  Output A (pipeline 2): data/organic_bipia_dataset.jsonl
    35k BIPIA-malicious rows (reused from data/final_training_dataset.jsonl) +
    35k organic benign rows (7k/source). Same 4 columns as the Alamsabi dataset.

  Output B (pipeline 3): data/organic_injected_dataset.jsonl
    70k organic rows (FeTaQA 10k, others 15k each). Half of each source's pool is
    injected using BIPIA's own insertion mechanism (insert_start/insert_middle/
    insert_end, reimplemented from microsoft/BIPIA's bipia/data/utils.py) and real
    attack instructions pulled from BIPIA's benchmark/text_attack_{train,test}.json.
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

try:
    from datasets import load_dataset, concatenate_datasets
except ImportError:
    print("❌ Error: Required library not installed. Please run: pip install datasets")
    exit(1)

# --- Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

PIPELINE2_N_PER_SOURCE = 7000

BIPIA_ATTACK_URLS = [
    "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_train.json",
    "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark/text_attack_test.json",
]

CNN_TEMPLATES = [
    "Summarize this article in a few sentences.",
    "What is the main news event described in this article?",
    "Write a short headline for this article.",
    "What are the key facts in this article?",
    "Explain what happened, based on this article.",
]


def ensure_nltk_punkt():
    """Make sure nltk's sentence tokenizer models are available for insert_middle."""
    import nltk
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


def usable_length(text: str, min_len: int, max_len: int) -> bool:
    return bool(text) and min_len <= len(text.strip()) <= max_len


# --- BIPIA Attack Pool ---
def fetch_attack_pool() -> list[tuple[str, str]]:
    """Fetch BIPIA's real attack-instruction pool: list of (instruction, category)."""
    pool = []
    for url in BIPIA_ATTACK_URLS:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for category, instructions in data.items():
            for instr in instructions:
                pool.append((instr, category))
    if not pool:
        raise RuntimeError("Failed to fetch any BIPIA attack instructions.")
    return pool


# --- BIPIA Insertion Functions (reimplemented from bipia/data/utils.py) ---
def insert_end(context: str, attack: str, rng: random.Random) -> str:
    return "\n".join([context, attack])


def insert_start(context: str, attack: str, rng: random.Random) -> str:
    return "\n".join([attack, context])


def insert_middle(context: str, attack: str, rng: random.Random) -> str:
    from nltk.tokenize import PunktSentenceTokenizer
    spans = list(PunktSentenceTokenizer().span_tokenize(context))
    if not spans:
        return insert_end(context, attack, rng)
    start, _ = rng.choice(spans)
    return "\n".join([context[:start], attack, context[start:]])


INSERT_FNS = {"start": insert_start, "middle": insert_middle, "end": insert_end}


def inject_row(context: str, rng: random.Random, attack_pool: list[tuple[str, str]]):
    instruction, category = rng.choice(attack_pool)
    position = rng.choice(list(INSERT_FNS.keys()))
    injected = INSERT_FNS[position](context, instruction, rng)
    return injected, instruction, position, category


# --- Per-Source Pull Functions ---
def pull_ms_marco(n: int, seed: int) -> list[dict]:
    print(f"  Pulling MS MARCO v2.1 (target {n:,} rows)...")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20000)
    rows = []
    for i, row in enumerate(tqdm(ds, desc="MS MARCO", total=n)):
        query = (row.get("query") or "").strip()
        passages = row.get("passages") or {}
        texts = passages.get("passage_text") or []
        sel = passages.get("is_selected") or []
        if not query or not texts:
            continue
        passage = next((t for t, s in zip(texts, sel) if s == 1), texts[0])
        passage = (passage or "").strip()
        if not usable_length(passage, 150, 6000):
            continue
        doc_id = f"ms_marco_{row.get('query_id', i)}"
        rows.append({"doc_id": doc_id, "context": passage, "user_intent": query})
        if len(rows) >= n:
            break
    return rows


def flatten_table(table_array) -> str:
    lines = [" | ".join(str(c) for c in r) for r in table_array]
    return "\n".join(lines)


def pull_fetaqa(n: int, seed: int) -> list[dict]:
    print(f"  Pulling FeTaQA (target {n:,} rows)...")
    ds = load_dataset("DongfuJiang/FeTaQA")
    combined = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    rows = []
    for i, row in enumerate(tqdm(combined, desc="FeTaQA")):
        question = (row.get("question") or "").strip()
        table_array = row.get("table_array")
        if not question or not table_array:
            continue
        table_text = flatten_table(table_array)
        if not usable_length(table_text, 50, 8000):
            continue
        doc_id = f"fetaqa_{row.get('feta_id', i)}"
        rows.append({"doc_id": doc_id, "context": table_text, "user_intent": question})
        if len(rows) >= n:
            break
    return rows


def pull_cosqa(n: int, seed: int) -> list[dict]:
    print(f"  Pulling CoSQA (target {n:,} rows)...")
    ds = load_dataset("gonglinyuan/CoSQA", split="train")
    rows = []
    for i, row in enumerate(tqdm(ds, desc="CoSQA")):
        code = (row.get("code") or "").strip()
        intent = (row.get("docstring_tokens") or "").strip()
        if not code or not intent or not usable_length(code, 20, 6000):
            continue
        doc_id = f"cosqa_{row.get('idx', i)}"
        rows.append({"doc_id": doc_id, "context": code, "user_intent": intent})
        if len(rows) >= n:
            break
    return rows


def pull_enron(n: int, seed: int) -> list[dict]:
    print(f"  Pulling Enron Email (target {n:,} rows)...")
    ds = load_dataset("corbt/enron-emails", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20000)
    rows = []
    for i, row in enumerate(tqdm(ds, desc="Enron", total=n)):
        body = (row.get("body") or "").strip()
        subject = (row.get("subject") or "").strip()
        if len(subject) < 3 or not usable_length(body, 200, 5000):
            continue
        doc_id = f"enron_{row.get('message_id', i)}"
        rows.append({"doc_id": doc_id, "context": body, "user_intent": subject})
        if len(rows) >= n:
            break
    return rows


def pull_cnn_dailymail(n: int, seed: int) -> list[dict]:
    print(f"  Pulling CNN/DailyMail (target {n:,} rows)...")
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20000)
    rows = []
    for i, row in enumerate(tqdm(ds, desc="CNN/DailyMail", total=n)):
        article = (row.get("article") or "").strip()[:6000]
        if not usable_length(article, 200, 6000):
            continue
        doc_id = f"cnn_dailymail_{row.get('id', i)}"
        user_intent = CNN_TEMPLATES[len(rows) % len(CNN_TEMPLATES)]
        rows.append({"doc_id": doc_id, "context": article, "user_intent": user_intent})
        if len(rows) >= n:
            break
    return rows


# --- Source Registry ---
SOURCES = {
    "ms_marco": {"display": "MS MARCO v2.1", "pipeline3_n": 15000, "pull": pull_ms_marco},
    "fetaqa": {"display": "FeTaQA", "pipeline3_n": 10000, "pull": pull_fetaqa},
    "cosqa": {"display": "CoSQA", "pipeline3_n": 15000, "pull": pull_cosqa},
    "enron": {"display": "Enron Email", "pipeline3_n": 15000, "pull": pull_enron},
    "cnn_dailymail": {"display": "CNN/DailyMail", "pipeline3_n": 15000, "pull": pull_cnn_dailymail},
}


# --- Dataset Builders ---
def build_pipeline3(pools: dict, rng: random.Random, attack_pool: list[tuple[str, str]]) -> list[dict]:
    rows_out = []
    for key, pool in pools.items():
        display = SOURCES[key]["display"]
        pool_ids = [r["doc_id"] for r in pool]
        half = len(pool_ids) // 2
        injected_ids = set(rng.sample(pool_ids, half)) if half else set()
        for r in pool:
            row = {
                "doc_id": r["doc_id"],
                "category": display,
                "context": r["context"],
                "user_intent": r["user_intent"],
                "original_context": r["context"],
                "label": 0,
                "attack_instruction": None,
                "injection_position": None,
                "attack_category": None,
            }
            if r["doc_id"] in injected_ids:
                injected_text, instruction, position, category = inject_row(r["context"], rng, attack_pool)
                row["context"] = injected_text
                row["label"] = 1
                row["attack_instruction"] = instruction
                row["injection_position"] = position
                row["attack_category"] = category
            rows_out.append(row)
    rng.shuffle(rows_out)
    return rows_out


def build_pipeline2(pools: dict, rng: random.Random, alamsabi_path: Path, pipeline2_n: int) -> list[dict]:
    print(f"  Loading Alamsabi BIPIA malicious rows from {alamsabi_path.name}...")
    alamsabi_df = pd.read_json(alamsabi_path, lines=True)
    bipia_df = alamsabi_df[alamsabi_df["source"] == "BIPIA"]
    malicious_rows = bipia_df[["context", "user_intent", "label", "source"]].to_dict("records")
    print(f"    - {len(malicious_rows):,} BIPIA malicious rows found")

    benign_rows = []
    for key, pool in pools.items():
        display = SOURCES[key]["display"]
        n = min(pipeline2_n, len(pool))
        sample = rng.sample(pool, n)
        for r in sample:
            benign_rows.append({
                "context": r["context"],
                "user_intent": r["user_intent"],
                "label": 0,
                "source": display,
            })
    print(f"    - {len(benign_rows):,} organic benign rows sampled ({pipeline2_n:,}/source)")

    all_rows = malicious_rows + benign_rows
    rng.shuffle(all_rows)
    return all_rows


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  ✔️ Wrote {len(rows):,} rows to {path}")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Build the organic-benign and organic-self-injected datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--limit", type=int, default=None, help="Pull only this many rows per source (for smoke testing).")
    parser.add_argument("--alamsabi-data", type=Path, default=DATA_DIR / "final_training_dataset.jsonl",
                         help="Path to the existing Alamsabi dataset (for BIPIA malicious rows).")
    args = parser.parse_args()

    if not args.alamsabi_data.is_file():
        print(f"❌ Error: Alamsabi dataset not found at: {args.alamsabi_data}")
        return

    ensure_nltk_punkt()
    rng = random.Random(args.seed)

    if args.limit:
        pipeline3_targets = {k: min(v["pipeline3_n"], args.limit) for k, v in SOURCES.items()}
        pipeline2_n = max(1, min(PIPELINE2_N_PER_SOURCE, args.limit // 2))
    else:
        pipeline3_targets = {k: v["pipeline3_n"] for k, v in SOURCES.items()}
        pipeline2_n = PIPELINE2_N_PER_SOURCE

    print("🔄 Fetching BIPIA attack instruction pool...")
    attack_pool = fetch_attack_pool()
    print(f"   - {len(attack_pool):,} attack instructions loaded")

    print("\n🚀 Pulling organic sources...")
    pools = {}
    for key, meta in SOURCES.items():
        n = pipeline3_targets[key]
        rows = meta["pull"](n, args.seed)
        if len(rows) < n:
            print(f"  ⚠️ Warning: only found {len(rows):,}/{n:,} usable rows for '{key}'")
        pools[key] = rows

    print("\n🧩 Building pipeline 3 dataset (organic + self-injected)...")
    pipeline3_rows = build_pipeline3(pools, rng, attack_pool)
    n_injected = sum(1 for r in pipeline3_rows if r["label"] == 1)
    print(f"   - {len(pipeline3_rows):,} total rows ({n_injected:,} injected, {len(pipeline3_rows) - n_injected:,} clean)")
    write_jsonl(pipeline3_rows, DATA_DIR / "organic_injected_dataset.jsonl")

    print("\n🧩 Building pipeline 2 dataset (organic benign + Alamsabi BIPIA malicious)...")
    pipeline2_rows = build_pipeline2(pools, rng, args.alamsabi_data, pipeline2_n)
    write_jsonl(pipeline2_rows, DATA_DIR / "organic_bipia_dataset.jsonl")

    print("\n🎉🎉🎉 Organic dataset construction completed! 🎉🎉🎉")


if __name__ == "__main__":
    main()
