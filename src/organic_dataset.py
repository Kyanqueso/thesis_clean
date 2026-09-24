import argparse
import json
import random
import sys
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
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_paths import DATA_DIR, DATASETS  # noqa: E402

PIPELINE2_N_PER_SOURCE = 7000

# BIPIA attack instruction pool is downloaded once and cached in the organic dataset's output directory.
BIPIA_RAW_BASE = "https://raw.githubusercontent.com/microsoft/BIPIA/main/benchmark"
ATTACK_FILES = {
    ("text", "train"): "text_attack_train.json",
    ("text", "test"): "text_attack_test.json",
    ("code", "train"): "code_attack_train.json",
    ("code", "test"): "code_attack_test.json",
}
ATTACK_SUBDIR = "attacks"

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


def _topup(rows: list[dict], dupes: list[dict], n: int, label: str) -> list[dict]:
    """Prefer distinct documents; if a source runs out of them, top up with
    repeats rather than returning short.

    Repeats are safe here because the train/test split groups rows by CONTENT,
    not by doc_id — two copies of the same text always land on the same side, so
    the model can never be tested on a document it trained on. CoSQA is the only
    source that needs this (~6.1k distinct functions against a 7k target).
    """
    if len(rows) < n and dupes:
        need = min(n - len(rows), len(dupes))
        print(f"  ↻ {label}: {len(rows):,} distinct documents; adding {need:,} "
              f"repeats to reach {n:,}")
        rows = rows + dupes[:need]
    return rows[:n]


# --- BIPIA Attack Pools ---
def load_attack_pools(attack_dir: Path, refresh: bool = False,
                      strict: bool = True) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """{(kind, part): [(instruction, attack_category), ...]} for kind in
    text/code and part in train/test.

    Each file is downloaded once into attack_dir and read from disk thereafter,
    so a rebuild needs no network and is pinned to the cached copy.

    `strict` drops from each TEST pool any attack category that also appears in
    the matching TRAIN pool. BIPIA ships exactly one such overlap on the text
    side ("Language Translation"); the code pools are already disjoint. This
    only matters when the pools are used separately (--disjoint-attacks).
    """
    attack_dir.mkdir(parents=True, exist_ok=True)
    pools: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for (kind, part), name in ATTACK_FILES.items():
        path = attack_dir / name
        if refresh or not path.is_file():
            resp = requests.get(f"{BIPIA_RAW_BASE}/{name}", timeout=30)
            resp.raise_for_status()
            path.write_bytes(resp.content)
            print(f"   ↓ cached {name} ({len(resp.content):,} bytes)")
        data = json.loads(path.read_text(encoding="utf-8"))
        pool = [(instr, category) for category, instrs in data.items() for instr in instrs]
        if not pool:
            raise RuntimeError(f"No BIPIA {kind}/{part} attack instructions in {attack_dir}")
        pools[(kind, part)] = pool

    if strict:
        for kind in ("text", "code"):
            train_cats = {c for _, c in pools[(kind, "train")]}
            kept = [(i, c) for i, c in pools[(kind, "test")] if c not in train_cats]
            n_dropped = len({c for _, c in pools[(kind, "test")]}) - len({c for _, c in kept})
            if n_dropped:
                print(f"   · {kind}: dropped {n_dropped} test attack type(s) also present in train")
            pools[(kind, "test")] = kept
    return pools


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
    rows, dupes, seen = [], [], set()
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
        entry = {"doc_id": f"ms_marco_{row.get('query_id', i)}",
                 "context": passage, "user_intent": query}
        if passage in seen:
            dupes.append(entry)
            continue
        seen.add(passage)
        rows.append(entry)
        if len(rows) >= n:
            break
    return _topup(rows, dupes, n, "MS MARCO v2.1")


def flatten_table(table_array) -> str:
    lines = [" | ".join(str(c) for c in r) for r in table_array]
    return "\n".join(lines)


def pull_fetaqa(n: int, seed: int) -> list[dict]:
    print(f"  Pulling FeTaQA (target {n:,} rows)...")
    ds = load_dataset("DongfuJiang/FeTaQA")
    combined = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
    rows, dupes, seen = [], [], set()
    for i, row in enumerate(tqdm(combined, desc="FeTaQA")):
        question = (row.get("question") or "").strip()
        table_array = row.get("table_array")
        if not question or not table_array:
            continue
        table_text = flatten_table(table_array)
        if not usable_length(table_text, 50, 8000):
            continue
        entry = {"doc_id": f"fetaqa_{row.get('feta_id', i)}",
                 "context": table_text, "user_intent": question}
        if table_text in seen:
            dupes.append(entry)
            continue
        seen.add(table_text)
        rows.append(entry)
        if len(rows) >= n:
            break
    return _topup(rows, dupes, n, "FeTaQA")


def pull_cosqa(n: int, seed: int) -> list[dict]:
    print(f"  Pulling CoSQA (target {n:,} rows)...")
    ds = load_dataset("gonglinyuan/CoSQA", split="train")
    rows, dupes, seen = [], [], set()
    for i, row in enumerate(tqdm(ds, desc="CoSQA")):
        code = (row.get("code") or "").strip()
        intent = (row.get("docstring_tokens") or "").strip()
        if not code or not intent or not usable_length(code, 20, 6000):
            continue
        entry = {"doc_id": f"cosqa_{row.get('idx', i)}",
                 "context": code, "user_intent": intent}
        if code in seen:
            dupes.append(entry)
            continue
        seen.add(code)
        rows.append(entry)
        if len(rows) >= n:
            break
    return _topup(rows, dupes, n, "CoSQA")


def pull_aeslc(n: int, seed: int, deidentify: bool = True) -> list[dict]:
    """Emails from AESLC (Zhang and Tetreault, ACL 2019) rather than the raw
    Enron corpus.

    Both are Enron mail, but AESLC is the curated subset: its authors kept only
    inbox/sent messages, stripped boilerplate the sender did not write, dropped
    replies and forwards so a body always matches its own subject, and required
    at least 3 sentences and 25 words. The raw corpus used before carried all of
    that — 46.6% of the rows it gave us were replies or forwards, some with a
    bare "RE:" as the user intent.

    AESLC is NOT de-identified at source: bodies still name people and carry
    phone numbers and addresses. Every email is therefore cleaned and scrubbed
    here, before dedup and before anything downstream sees it, so no raw email
    text reaches data/, the embeddings or the dashboard. `deidentify=False` is
    for the audit tool only, which needs the before/after pair.

    18,302 emails over the three splits, which are combined here: the subject
    line generation split is irrelevant to injection detection, and one pool of
    ~15.6k usable emails covers the 7,000 target. doc_ids are positional, so
    they carry no pointer back to a source record.
    """
    print(f"  Pulling AESLC email (target {n:,} rows)...")
    ds = load_dataset("Yale-LILY/aeslc")
    combined = concatenate_datasets([ds[s] for s in ("train", "validation", "test")])
    # small enough to index directly; shuffle the order so the pool is drawn
    # across all three splits rather than filling up from train alone
    order = list(range(len(combined)))
    random.Random(seed).shuffle(order)

    # Raw candidates first — filtering is cheap, de-identification is not, and
    # the name vocabulary has to see the whole pool at once. The margin covers
    # the few emails that fall under the length floor once a disclaimer is cut.
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

    if deidentify:
        from anonymize_enron import scrub_pool
        print(f"   de-identifying {len(candidates):,} emails "
              f"(Presidio + Enron recognizers + pool name vocabulary)...")
        candidates, vocab_size = scrub_pool(candidates)
        print(f"   name vocabulary: {vocab_size:,} names swept over every email")

    rows, dupes, seen = [], [], set()
    for body, subject in candidates:
        # cutting a disclaimer can take a body under the floor, and a subject
        # that was only a name is now only a placeholder
        if len(subject) < 3 or not usable_length(body, 200, 5000):
            continue
        # dedup on the text we will actually keep: two emails differing only in
        # the names they mention are the same document once scrubbed
        entry = {"doc_id": None, "context": body, "user_intent": subject}
        if body in seen:
            dupes.append(entry)
            continue
        seen.add(body)
        rows.append(entry)
        if len(rows) >= n:
            break
    rows = _topup(rows, dupes, n, "AESLC email")
    for k, row in enumerate(rows):
        row["doc_id"] = f"aeslc_{k:05d}"
    return rows


def pull_cnn_dailymail(n: int, seed: int) -> list[dict]:
    print(f"  Pulling CNN/DailyMail (target {n:,} rows)...")
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=20000)
    rows, dupes, seen = [], [], set()
    for i, row in enumerate(tqdm(ds, desc="CNN/DailyMail", total=n)):
        article = (row.get("article") or "").strip()[:6000]
        if not usable_length(article, 200, 6000):
            continue
        entry = {"doc_id": f"cnn_dailymail_{row.get('id', i)}", "context": article,
                 "user_intent": CNN_TEMPLATES[len(rows) % len(CNN_TEMPLATES)]}
        if article in seen:
            dupes.append(entry)
            continue
        seen.add(article)
        rows.append(entry)
        if len(rows) >= n:
            break
    return _topup(rows, dupes, n, "CNN/DailyMail")


# --- Source Datasets (Organic) ---
# attack_kind picks which BIPIA pool a source's injections come from: prose gets
# text attacks, CoSQA gets BIPIA's code attacks (a text attack appended to a
# Python function is not what a code-assistant injection looks like).
# pipeline3_n is a DOCUMENT target: each document becomes two rows (clean +
# injected), so 7,000 docs/source -> 14,000 rows/source. Pools prefer distinct
# documents and only repeat when a source runs dry — CoSQA has ~6,130 distinct
# functions, so ~870 of its 7,000 are repeats (see _topup).
SOURCES = {
    "ms_marco": {"display": "MS MARCO v2.1", "pipeline3_n": 7000, "pull": pull_ms_marco, "attack_kind": "text"},
    "fetaqa": {"display": "FeTaQA", "pipeline3_n": 7000, "pull": pull_fetaqa, "attack_kind": "text"},
    "cosqa": {"display": "CoSQA", "pipeline3_n": 7000, "pull": pull_cosqa, "attack_kind": "code"},
    "aeslc": {"display": "AESLC Email", "pipeline3_n": 7000, "pull": pull_aeslc, "attack_kind": "text"},
    "cnn_dailymail": {"display": "CNN/DailyMail", "pipeline3_n": 7000, "pull": pull_cnn_dailymail, "attack_kind": "text"},
}


# --- Dataset Builders ---
def _p3_row(r: dict, display: str, context: str, label: int, instruction=None,
            position=None, category=None, split=None) -> dict:
    """One pipeline-3 row. A document's two rows share doc_id, user_intent and
    original_context; only `context` and the attack fields differ. `split` is
    only written under --disjoint-attacks, where train/test must be decided at
    build time so each side can draw from its own attack pool."""
    row = {
        "doc_id": r["doc_id"],
        "category": display,
        "context": context,
        "user_intent": r["user_intent"],
        "original_context": r["context"],
        "label": label,
        "attack_instruction": instruction,
        "injection_position": position,
        "attack_category": category,
    }
    if split is not None:
        row["split"] = split
    return row


def build_pipeline3(pools: dict, rng: random.Random, attack_pools: dict,
                    disjoint_attacks: bool = True,
                    train_frac: float = 0.8) -> list[dict]:
    """Twin design: every document is used twice, once clean and once injected.

    Because the same document carries both labels, its length, topic, vocabulary
    and intent are balanced across the classes exactly rather than on average —
    the injected instruction is the only thing a classifier can key on.
    7,000 documents/source -> 14,000 rows/source.

    disjoint_attacks decides where the injections come from:

    - True (default): documents are assigned train/test here, and each side
      draws only from its own BIPIA pool (25 train types vs 24 test types, zero
      overlap). The assignment is written to a `split` column that
      train_evaluate_visualize.py honors, so the run measures generalization to
      attack types the model has NEVER seen — the deployment condition.
    - False (--merged-attacks): every document draws from BIPIA's train AND test
      pools combined, so test attacks are also seen during training. Only useful
      as the "known attacks" comparison run.
    """
    rows_out = []
    for key, pool in pools.items():
        display = SOURCES[key]["display"]
        kind = SOURCES[key]["attack_kind"]
        if disjoint_attacks:
            attack_for = {p: attack_pools[(kind, p)] for p in ("train", "test")}
        else:
            merged = attack_pools[(kind, "train")] + attack_pools[(kind, "test")]
            attack_for = {"train": merged, "test": merged}

        docs = list(pool)
        rng.shuffle(docs)
        # Assign train/test per DISTINCT text, not per doc_id: a pool topped up
        # with repeats holds the same text under two ids, and both copies must
        # land on the same side or the model is tested on what it trained on.
        texts = list(dict.fromkeys(r["context"] for r in docs))
        n_train = round(len(texts) * train_frac)
        part_of = {t: ("train" if i < n_train else "test") for i, t in enumerate(texts)}
        for r in docs:
            part = part_of[r["context"]]
            split = part if disjoint_attacks else None
            injected, instruction, position, category = inject_row(
                r["context"], rng, attack_for[part])
            rows_out.append(_p3_row(r, display, r["context"], 0, split=split))
            rows_out.append(_p3_row(r, display, injected, 1, instruction,
                                    position, category, split=split))
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
            print(f"    ⚠️ {display}: only {n:,} unique documents (wanted {pipeline2_n:,})")
        for r in rng.sample(pool, n):
            benign_rows.append({
                "context": r["context"],
                "user_intent": r["user_intent"],
                "label": 0,
                "source": display,
            })
    print(f"    - {len(benign_rows):,} organic benign rows sampled (target {pipeline2_n:,}/source)")

    # Balance 1:1 by downsampling the malicious side. Never pad the benign side,
    # which would mean repeating a document — exactly the duplication the pools
    # are deduplicated to avoid.
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
    print(f"  ✔️ Wrote {len(rows):,} rows to {path}")


# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(
        description="Build the organic-benign and organic-self-injected datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--limit", type=int, default=None, help="Pull only this many rows per source (for smoke testing).")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR,
                         help="Directory to write the built datasets into.")
    parser.add_argument("--alamsabi-data", type=Path, default=None,
                         help="Path to the existing Alamsabi dataset (default: <out-dir>/%s)."
                              % DATASETS["pipeline1_alamsabi"])
    parser.add_argument("--refresh-attacks", action="store_true",
                         help="Re-download BIPIA's attack pools instead of reusing the cached copies.")
    parser.add_argument("--merged-attacks", action="store_true",
                         help="Pipeline 3: draw injections from BIPIA's train AND test attack pools "
                              "combined, so test attack types are also seen during training. The "
                              "default holds the test pool out; use this only to build the "
                              "'known attacks' comparison run.")
    parser.add_argument("--train-frac", type=float, default=0.8,
                         help="Pipeline 3: fraction of documents assigned to train at build time.")
    args = parser.parse_args()
    disjoint_attacks = not args.merged_attacks

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.alamsabi_data is None:
        args.alamsabi_data = out_dir / DATASETS["pipeline1_alamsabi"]

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

    print("🔄 Loading BIPIA attack instruction pools...")
    attack_pools = load_attack_pools(out_dir / ATTACK_SUBDIR, refresh=args.refresh_attacks)
    for (kind, part), pool in sorted(attack_pools.items()):
        print(f"   - {kind}/{part}: {len(pool):,} instructions, "
              f"{len({c for _, c in pool})} attack types")
    if disjoint_attacks:
        print("   ⚡ held-out attacks (default): test documents get attack types never seen in training")
    else:
        print("   ⚠️ --merged-attacks: test attack types are ALSO seen during training")

    print("\n🚀 Pulling organic sources...")
    pools = {}
    for key, meta in SOURCES.items():
        n = pipeline3_targets[key]
        rows = meta["pull"](n, args.seed)
        if len(rows) < n:
            print(f"  ⚠️ Warning: only found {len(rows):,}/{n:,} usable rows for '{key}'")
        pools[key] = rows

    print("\n🧩 Building pipeline 3 dataset (organic, each document clean + injected)...")
    pipeline3_rows = build_pipeline3(pools, rng, attack_pools,
                                     disjoint_attacks=disjoint_attacks,
                                     train_frac=args.train_frac)
    n_injected = sum(1 for r in pipeline3_rows if r["label"] == 1)
    n_docs = len({r["doc_id"] for r in pipeline3_rows})
    print(f"   - {len(pipeline3_rows):,} rows from {n_docs:,} documents "
          f"({n_injected:,} injected, {len(pipeline3_rows) - n_injected:,} clean)")
    if disjoint_attacks:
        tr = {r["attack_category"] for r in pipeline3_rows if r["split"] == "train" and r["label"] == 1}
        te = {r["attack_category"] for r in pipeline3_rows if r["split"] == "test" and r["label"] == 1}
        print(f"   - attack types: {len(tr)} train / {len(te)} test, "
              f"{len(tr & te)} shared  (0 = test attacks are unseen)")
    write_jsonl(pipeline3_rows, out_dir / DATASETS["pipeline3_organic_injected"])

    print("\n🧩 Building pipeline 2 dataset (organic benign + Alamsabi BIPIA malicious)...")
    pipeline2_rows = build_pipeline2(pools, rng, args.alamsabi_data, pipeline2_n)
    write_jsonl(pipeline2_rows, out_dir / DATASETS["pipeline2_organic_bipia"])

    print("\n🎉🎉🎉 Organic dataset construction completed! 🎉🎉🎉")


if __name__ == "__main__":
    main()
