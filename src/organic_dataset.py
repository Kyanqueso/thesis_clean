import json
import random
import sys
from pathlib import Path

import nltk
import pandas as pd
import requests
from datasets import concatenate_datasets, load_dataset
from nltk.tokenize import PunktSentenceTokenizer
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# deidentify imports this module back for its audit; plain module imports keep that cycle safe
import deidentify  # noqa: E402
import progress  # noqa: E402  (src/progress.py: the marker lines the dashboard's bars read)
from pipeline_paths import DATA_DIR, DATASETS  # noqa: E402

SEED = 42
N_PER_SOURCE = 7000   # documents per organic source, in both pipeline 2 and pipeline 3
TRAIN_FRAC = 0.8   # pipeline 3: share of documents assigned to train at build time
ALAMSABI_PATH = DATA_DIR / DATASETS["pipeline1_alamsabi"]

# BIPIA attack pools are downloaded once and cached under data/attacks
BIPIA_RAW_BASE = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark"
ATTACK_FILES = {
    ("text", "train"): "text_attack_train.json",
    ("text", "test"): "text_attack_test.json",
    ("code", "train"): "code_attack_train.json",
    ("code", "test"): "code_attack_test.json",
}
ATTACK_SUBDIR = "attacks"

# CNN/DailyMail has no user question, so every article gets this one
CNN_INTENT = "Summarize the following text"


def ensure_nltk_punkt():
    """Make sure nltk's sentence tokenizer models are available for insert_middle."""
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


def _collect(entries, n: int, label: str) -> list[dict]:
    """First n entries with distinct context, topped up with repeats if a source runs short."""
    rows, dupes, seen = [], [], set()
    for entry in entries:
        if entry["context"] in seen:
            dupes.append(entry)
            continue
        seen.add(entry["context"])
        rows.append(entry)
        if len(rows) >= n:
            break
    # repeats are safe: the train/test split groups rows by text, so copies stay on one side
    if len(rows) < n and dupes:
        need = min(n - len(rows), len(dupes))
        print(f"  ↻ {label}: {len(rows):,} distinct documents; adding {need:,} "
              f"repeats to reach {n:,}")
        rows = rows + dupes[:need]
    return rows[:n]


def load_attack_pools(attack_dir: Path) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """{(kind, part): [(instruction, category)]}, with test categories also in train dropped."""
    attack_dir.mkdir(parents=True, exist_ok=True)
    pools: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for (kind, part), name in ATTACK_FILES.items():
        path = attack_dir / name
        if not path.is_file():
            resp = requests.get(f"{BIPIA_RAW_BASE}/{name}", timeout=30)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            print(f"   ↓ cached {name} ({len(resp.content):,} bytes)")
        data = json.loads(path.read_text(encoding="utf-8"))
        pool = [(instr, category) for category, instrs in data.items() for instr in instrs]
        if not pool:
            raise RuntimeError(f"No BIPIA {kind}/{part} attack instructions in {attack_dir}")
        pools[(kind, part)] = pool

    for kind in ("text", "code"):
        train_cats = {c for _, c in pools[(kind, "train")]}
        kept = [(i, c) for i, c in pools[(kind, "test")] if c not in train_cats]
        n_dropped = len({c for _, c in pools[(kind, "test")]}) - len({c for _, c in kept})
        if n_dropped:
            print(f"   · {kind}: dropped {n_dropped} test attack type(s) also present in train")
        pools[(kind, "test")] = kept
    return pools


# BIPIA insertion functions, reimplemented from bipia/data/utils.py
def insert_end(context: str, attack: str, rng: random.Random) -> str:
    return "\n".join([context, attack])


def insert_start(context: str, attack: str, rng: random.Random) -> str:
    return "\n".join([attack, context])


def insert_middle(context: str, attack: str, rng: random.Random) -> str:
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


def pull_ms_marco(n: int, seed: int) -> list[dict]:
    print(f"  Pulling MS MARCO v2.1 (target {n:,} rows)...")
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20000)

    def entries():
        for i, row in enumerate(tqdm(ds, desc="MS MARCO", total=n)):
            query = (row.get("query") or "").strip()
            passages = row.get("passages") or {}
            texts = passages.get("passage_text") or []
            sel = passages.get("is_selected") or []
            if not query or not texts:
                continue
            passage = next((t for t, s in zip(texts, sel) if s == 1), texts[0])
            passage = (passage or "").strip()
            if usable_length(passage, 150, 6000):
                yield {"doc_id": f"ms_marco_{row.get('query_id', i)}",
                       "context": passage, "user_intent": query}

    return _collect(entries(), n, "MS MARCO v2.1")


def flatten_table(table_array) -> str:
    lines = [" | ".join(str(c) for c in r) for r in table_array]
    return "\n".join(lines)


def pull_fetaqa(n: int, seed: int) -> list[dict]:
    print(f"  Pulling FeTaQA (target {n:,} rows)...")
    ds = load_dataset("DongfuJiang/FeTaQA")
    combined = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])

    def entries():
        for i, row in enumerate(tqdm(combined, desc="FeTaQA")):
            question = (row.get("question") or "").strip()
            table_array = row.get("table_array")
            if not question or not table_array:
                continue
            table_text = flatten_table(table_array)
            if usable_length(table_text, 50, 8000):
                yield {"doc_id": f"fetaqa_{row.get('feta_id', i)}",
                       "context": table_text, "user_intent": question}

    return _collect(entries(), n, "FeTaQA")


def pull_cosqa(n: int, seed: int) -> list[dict]:
    print(f"  Pulling CoSQA (target {n:,} rows)...")
    ds = load_dataset("gonglinyuan/CoSQA", split="train")

    def entries():
        for i, row in enumerate(tqdm(ds, desc="CoSQA")):
            code = (row.get("code") or "").strip()
            # the web query the code was retrieved for; some are wrapped in quotes
            intent = (row.get("doc") or "").strip().strip('"').strip()
            if code and intent and usable_length(code, 20, 6000):
                yield {"doc_id": f"cosqa_{row.get('idx', i)}",
                       "context": code, "user_intent": intent}

    return _collect(entries(), n, "CoSQA")


def pull_aeslc(n: int, seed: int, scrub: bool = True) -> list[dict]:

    print(f"  Pulling AESLC email (target {n:,} rows)...")
    ds = load_dataset("Yale-LILY/aeslc")
    combined = concatenate_datasets([ds[s] for s in ("train", "validation", "test")])
    # shuffle so the pool draws from all three splits, not from train alone
    order = list(range(len(combined)))
    random.Random(seed).shuffle(order)

    # collect raw candidates first: the name vocabulary needs the whole pool at once
    candidates = []
    for i in tqdm(order, desc="AESLC"):
        row = combined[i]
        body = (row.get("email_body") or "").strip()
        subject = (row.get("subject_line") or "").strip()
        if len(subject) < 3 or not usable_length(body, 200, 5000):
            continue
        candidates.append((body, subject))
        if len(candidates) >= int(n * 1.2) + 50:
            break

    if scrub:
        print(f"   de-identifying {len(candidates):,} emails "
              f"(Presidio + custom recognizers + pool name vocabulary)...")
        candidates, vocab_size = deidentify.scrub_pool(candidates)
        print(f"   name vocabulary: {vocab_size:,} names swept over every email")

    # re-check lengths (scrubbing can push a body under the floor) and dedup on the scrubbed text
    rows = _collect(({"doc_id": None, "context": body, "user_intent": subject}
                     for body, subject in candidates
                     if len(subject) >= 3 and usable_length(body, 200, 5000)),
                    n, "AESLC email")
    for k, row in enumerate(rows):
        row["doc_id"] = f"aeslc_{k:05d}"
    return rows


def pull_cnn_dailymail(n: int, seed: int) -> list[dict]:
    print(f"  Pulling CNN/DailyMail (target {n:,} rows)...")
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20000)

    def entries():
        for i, row in enumerate(tqdm(ds, desc="CNN/DailyMail", total=n)):
            article = (row.get("article") or "").strip()[:6000]
            if usable_length(article, 200, 6000):
                yield {"doc_id": f"cnn_dailymail_{row.get('id', i)}", "context": article,
                       "user_intent": CNN_INTENT}

    return _collect(entries(), n, "CNN/DailyMail")


SOURCES = {
    "ms_marco": {"display": "MS MARCO v2.1", "pull": pull_ms_marco, "attack_kind": "text"},
    "fetaqa": {"display": "FeTaQA", "pull": pull_fetaqa, "attack_kind": "text"},
    "cosqa": {"display": "CoSQA", "pull": pull_cosqa, "attack_kind": "code"},
    "aeslc": {"display": "AESLC Email", "pull": pull_aeslc, "attack_kind": "text"},
    "cnn_dailymail": {"display": "CNN/DailyMail", "pull": pull_cnn_dailymail, "attack_kind": "text"},
}


def _p3_row(r: dict, display: str, context: str, label: int, split: str,
            instruction=None, position=None, category=None) -> dict:
    return {
        "doc_id": r["doc_id"],
        "category": display,
        "context": context,
        "user_intent": r["user_intent"],
        "original_context": r["context"],
        "label": label,
        "attack_instruction": instruction,
        "injection_position": position,
        "attack_category": category,
        "split": split,
    }


def build_pipeline3(pools: dict, rng: random.Random, attack_pools: dict) -> list[dict]:
    """Twin design: every document appears once clean and once injected."""
    rows_out = []
    for key, pool in pools.items():
        display = SOURCES[key]["display"]
        kind = SOURCES[key]["attack_kind"]
        docs = list(pool)
        rng.shuffle(docs)
        # split by distinct text, not doc_id, so repeated documents land on the same side
        texts = list(dict.fromkeys(r["context"] for r in docs))
        n_train = round(len(texts) * TRAIN_FRAC)
        part_of = {t: ("train" if i < n_train else "test") for i, t in enumerate(texts)}
        for r in docs:
            part = part_of[r["context"]]
            # train and test draw from separate BIPIA pools, so test attack types are unseen
            injected, instruction, position, category = inject_row(
                r["context"], rng, attack_pools[(kind, part)])
            rows_out.append(_p3_row(r, display, r["context"], 0, part))
            rows_out.append(_p3_row(r, display, injected, 1, part,
                                    instruction, position, category))
    rng.shuffle(rows_out)
    return rows_out


def build_pipeline2(pools: dict, rng: random.Random, alamsabi_path: Path, pipeline2_n: int) -> list[dict]:
    print(f"  Loading Alamsabi BIPIA malicious rows from {alamsabi_path.name}...")
    alamsabi_df = pd.read_json(alamsabi_path, lines=True)
    bipia_df = alamsabi_df[alamsabi_df["source"] == "BIPIA"]
    malicious_rows = bipia_df[["context", "user_intent", "label", "source"]].to_dict("records")
    print(f"    - {len(malicious_rows):,} BIPIA malicious rows available")

    benign_rows = []
    for key, pool in pools.items():
        display = SOURCES[key]["display"]
        n = min(pipeline2_n, len(pool))
        if n < pipeline2_n:
            print(f"    {display}: only {n:,} unique documents (wanted {pipeline2_n:,})")
        for r in rng.sample(pool, n):
            benign_rows.append({
                "context": r["context"],
                "user_intent": r["user_intent"],
                "label": 0,
                "source": display,
            })
    print(f"    - {len(benign_rows):,} organic benign rows sampled (target {pipeline2_n:,}/source)")

    # balance 1:1 by downsampling the malicious side, never by repeating benign documents
    if len(malicious_rows) > len(benign_rows):
        print(f"    - downsampling malicious {len(malicious_rows):,} -> "
              f"{len(benign_rows):,} for a 1:1 balance")
        malicious_rows = rng.sample(malicious_rows, len(benign_rows))

    all_rows = malicious_rows + benign_rows
    rng.shuffle(all_rows)
    return all_rows


def write_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):,} rows to {path}")


def main() -> int:
    out_dir = DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ALAMSABI_PATH.is_file():
        print(f"Error: Alamsabi dataset not found at: {ALAMSABI_PATH} (run download_dataset.py first)")
        return 1

    ensure_nltk_punkt()
    rng = random.Random(SEED)

    print("Loading BIPIA attack instruction pools...")
    attack_pools = load_attack_pools(out_dir / ATTACK_SUBDIR)
    for (kind, part), pool in sorted(attack_pools.items()):
        print(f"   - {kind}/{part}: {len(pool):,} instructions, "
              f"{len({c for _, c in pool})} attack types")
    print("   held-out attacks: test documents get attack types never seen in training")

    print("\nPulling organic sources...")
    pools = {}
    for k, (key, meta) in enumerate(SOURCES.items()):
        progress.mark("pull", meta["display"], k, len(SOURCES))
        rows = meta["pull"](N_PER_SOURCE, SEED)
        if len(rows) < N_PER_SOURCE:
            print(f"  Warning: only found {len(rows):,}/{N_PER_SOURCE:,} usable rows for '{key}'")
        pools[key] = rows

    progress.mark("build", "pipeline 3 (organic, self-injected)", 0, 2)
    print("\nBuilding pipeline 3 dataset (organic, each document clean + injected)...")
    pipeline3_rows = build_pipeline3(pools, rng, attack_pools)
    n_injected = sum(1 for r in pipeline3_rows if r["label"] == 1)
    n_docs = len({r["doc_id"] for r in pipeline3_rows})
    print(f"   - {len(pipeline3_rows):,} rows from {n_docs:,} documents "
          f"({n_injected:,} injected, {len(pipeline3_rows) - n_injected:,} clean)")
    tr = {r["attack_category"] for r in pipeline3_rows if r["split"] == "train" and r["label"] == 1}
    te = {r["attack_category"] for r in pipeline3_rows if r["split"] == "test" and r["label"] == 1}
    print(f"   - attack types: {len(tr)} train / {len(te)} test, "
          f"{len(tr & te)} shared  (0 = test attacks are unseen)")
    write_jsonl(pipeline3_rows, out_dir / DATASETS["pipeline3_organic_injected"])

    progress.mark("build", "pipeline 2 (organic + BIPIA)", 1, 2)
    print("\nBuilding pipeline 2 dataset (organic benign + Alamsabi BIPIA malicious)...")
    pipeline2_rows = build_pipeline2(pools, rng, ALAMSABI_PATH, N_PER_SOURCE)
    write_jsonl(pipeline2_rows, out_dir / DATASETS["pipeline2_organic_bipia"])

    print("\nOrganic dataset construction completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
