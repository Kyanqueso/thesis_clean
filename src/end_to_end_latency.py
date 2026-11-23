"""Time one prompt end to end (embed, then classify) for every pipeline, embedding and saved classifier."""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline_paths import DATA_DIR, DATASETS, MODES, RUNS_DIR  # noqa: E402

SEPARATOR = "\n\n---\n\n"          # must match src/generate_embeddings.py
OPENAI_MAX_TOKENS = 8191           # same truncation as src/generate_embeddings.py
EMBEDDINGS = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "qwen3": "Qwen/Qwen3-Embedding-4B",
    "openai": "text-embedding-3-small",
}
CLASSIFIERS = ["LogisticRegression", "SVM", "XGBoost", "LightGBM", "RandomForest"]


def describe(exc: Exception) -> str:
    """Error text without any API key fragment (OpenAI echoes part of a rejected key)."""
    return f"{type(exc).__name__}: " + re.sub(r"sk-[\w*-]+", "sk-…", str(exc))[:300]


def build_text(intent: str, context: str, mode: str) -> str:
    if mode == "user_intent_only":
        return intent
    if mode == "context_only":
        return context
    return f"{intent}{SEPARATOR}{context}"


class Embedder:
    """One request at a time, preprocessed exactly as in training."""

    def __init__(self, name: str, device: str):
        self.name = name
        t0 = time.perf_counter()
        if name == "openai":
            import tiktoken
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv(ROOT / ".env")
            self.client = OpenAI()   # raises if OPENAI_API_KEY is missing
            self.tok = tiktoken.encoding_for_model(EMBEDDINGS[name])
            self.device = "OpenAI API"
        else:
            import torch
            from sentence_transformers import SentenceTransformer
            self.torch = torch
            kwargs = {"model_kwargs": {"torch_dtype": torch.float16}} if name == "qwen3" else {}
            self.model = SentenceTransformer(EMBEDDINGS[name], device=device, **kwargs)
            if name == "qwen3":
                self.model.max_seq_length = 2048
            self.device = str(self.model.device)
        self.load_s = time.perf_counter() - t0

    def __call__(self, text: str) -> np.ndarray:
        if self.name == "openai":
            text = text if text.strip() else " "
            ids = self.tok.encode(text)
            if len(ids) > OPENAI_MAX_TOKENS:
                text = self.tok.decode(ids[:OPENAI_MAX_TOKENS])
            resp = self.client.embeddings.create(model=EMBEDDINGS["openai"], input=[text])
            return np.asarray(resp.data[0].embedding, dtype=np.float32)
        vec = self.model.encode([text], batch_size=1, show_progress_bar=False, normalize_embeddings=False)
        return np.asarray(vec[0])   # encode returns a CPU array, so GPU work has finished here


def pick_rows(run: Path, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """n test rows of the run, half benign and half malicious, with their stored probabilities."""
    pred = np.load(run / "results" / "predictions.npz")
    test_idx, y = pred["test_idx"].astype(np.int64), pred["y_true"].astype(int)
    rng = np.random.default_rng(seed)
    pos = np.concatenate([rng.choice(np.flatnonzero(y == c), n // 2, replace=False) for c in (0, 1)])
    rng.shuffle(pos)
    stored = {k: pred[k][pos] for k in pred.files if "__" in k}
    return test_idx[pos], y[pos], stored


def summarise(req: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pipe, emb, clf), g in req.groupby(["pipeline", "embedding", "classifier"], sort=False):
        e2e, emb_ms, clf_ms = g.e2e_ms.to_numpy(), g.embed_ms.to_numpy(), g.classify_ms.to_numpy()
        rows.append({
            "Pipeline": pipe, "Embedding": emb.upper(), "Classifier": clf, "Requests": len(g),
            "Embed-P50(ms)": np.percentile(emb_ms, 50), "Embed-Mean(ms)": emb_ms.mean(),
            "Classify-P50(ms)": np.percentile(clf_ms, 50), "Classify-Mean(ms)": clf_ms.mean(),
            "E2E-P50(ms)": np.percentile(e2e, 50), "E2E-P95(ms)": np.percentile(e2e, 95),
            "E2E-P99(ms)": np.percentile(e2e, 99), "E2E-Mean(ms)": e2e.mean(), "E2E-Max(ms)": e2e.max(),
            "Max-Prob-Diff": g.prob_diff.max(),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipelines", default="all",
                    help=f"comma-separated subset of {', '.join(DATASETS)}, or all (default)")
    ap.add_argument("--mode", default="normal", choices=MODES)
    ap.add_argument("--embeddings", default="minilm,qwen3,openai",
                    help="comma-separated subset of minilm, qwen3, openai")
    ap.add_argument("--n", type=int, default=200, help="timed requests per embedding (even number)")
    ap.add_argument("--warmup", type=int, default=10, help="untimed requests first, per embedding")
    ap.add_argument("--device", default=None, help="cuda or cpu for the local models; default: cuda if present")
    ap.add_argument("--seed", type=int, default=42)
    # writes end_to_end_latency.csv (percentiles per embedding x classifier), _requests.csv (one row per
    # timed request) and _meta.json (machine, versions, settings, model load times)
    ap.add_argument("--out", type=Path, default=RUNS_DIR / "analysis")
    args = ap.parse_args()

    embs = [e.strip() for e in args.embeddings.split(",") if e.strip()]
    if unknown := [e for e in embs if e not in EMBEDDINGS]:
        ap.error(f"unknown embedding(s): {', '.join(unknown)}")
    if args.n < 2 or args.n % 2:
        ap.error("--n must be an even number of at least 2")
    pipes = list(DATASETS) if args.pipelines == "all" else [p.strip() for p in args.pipelines.split(",") if p.strip()]
    if unknown := [p for p in pipes if p not in DATASETS]:
        ap.error(f"unknown pipeline(s): {', '.join(unknown)}")

    jobs = []
    for pipe in pipes:
        run = RUNS_DIR / pipe / args.mode
        if not (run / "results" / "predictions.npz").is_file():
            print(f"{pipe}/{args.mode}: no finished run; skipped")
            continue
        df = pd.read_json(DATA_DIR / DATASETS[pipe], lines=True)
        rows, labels, stored = pick_rows(run, args.n + args.warmup, args.seed)
        texts = [build_text(str(df.user_intent.iat[i]), str(df.context.iat[i]), args.mode) for i in rows]
        jobs.append((pipe, run, texts, labels, stored))
        print(f"{pipe}/{args.mode}: {args.warmup} warm-up + {args.n} timed requests per embedding, "
              f"drawn from the {len(df):,}-row dataset's test split")

    import joblib
    records, meta_emb = [], {}
    for emb in embs:
        print(f"\n[{emb}] loading {EMBEDDINGS[emb]} ...")
        try:
            embedder = Embedder(emb, args.device)
        except Exception as exc:   # missing key, no GPU memory, missing package: skip this one
            print(f"[{emb}] could not load: {describe(exc)}")
            meta_emb[emb] = {"error": f"{describe(exc)}"}
            continue
        print(f"[{emb}] loaded on {embedder.device} in {embedder.load_s:.1f} s")
        meta_emb[emb] = {"device": embedder.device, "load_s": round(embedder.load_s, 2)}

        for pipe, run, texts, labels, stored in jobs:
            models = {c: joblib.load(run / "models" / f"{emb}_{c}.joblib") for c in CLASSIFIERS
                      if (run / "models" / f"{emb}_{c}.joblib").is_file()}
            if not models:
                print(f"[{pipe} · {emb}] no saved classifiers in {run / 'models'}; skipped")
                continue
            try:
                records += time_embedding(pipe, emb, embedder, models, texts, labels, stored, args)
            except Exception as exc:   # e.g. a rejected API key or a network failure: keep the others
                print(f"[{pipe} · {emb}] failed while timing, results discarded: {describe(exc)}")
                meta_emb[emb].setdefault("errors", {})[pipe] = describe(exc)
        del embedder

    if not records:
        print("\nNothing was timed.")
        return 1
    return report(records, meta_emb, args)


def time_embedding(pipe, emb, embedder, models, texts, labels, stored, args) -> list[dict]:
    warm, timed = slice(0, args.warmup), slice(args.warmup, None)
    for text in texts[warm]:   # first calls allocate memory, compile kernels, open connections
        vec = embedder(text).reshape(1, -1)
        for m in models.values():
            m.predict_proba(vec)

    records = []
    for k, (text, y) in enumerate(zip(texts[timed], labels[timed])):
        t0 = time.perf_counter()
        vec = embedder(text).reshape(1, -1)
        embed_ms = (time.perf_counter() - t0) * 1000
        for clf, m in models.items():
            t1 = time.perf_counter()
            prob = float(m.predict_proba(vec)[0, 1])
            clf_ms = (time.perf_counter() - t1) * 1000
            ref = stored.get(f"{emb}__{clf}")
            records.append({
                "pipeline": pipe, "embedding": emb, "classifier": clf, "request": k, "label": int(y),
                "embed_ms": embed_ms, "classify_ms": clf_ms, "e2e_ms": embed_ms + clf_ms,
                "probability": prob,
                "stored_probability": float(ref[args.warmup + k]) if ref is not None else np.nan,
            })
        if (k + 1) % 50 == 0 or k + 1 == args.n:
            print(f"[{pipe} · {emb}] {k + 1}/{args.n} requests")
    return records


def report(records: list[dict], meta_emb: dict, args) -> int:
    req = pd.DataFrame(records)
    req["prob_diff"] = (req.probability - req.stored_probability).abs()
    summary = summarise(req)
    args.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out / "end_to_end_latency.csv", index=False)
    req.to_csv(args.out / "end_to_end_latency_requests.csv", index=False)

    gpu = None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    meta = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "pipelines": list(dict.fromkeys(req.pipeline)), "mode": args.mode, "n_timed": args.n, "n_warmup": args.warmup,
        "seed": args.seed, "batch_size": 1, "timer": "time.perf_counter",
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "python": platform.python_version(), "gpu": gpu},
        "embeddings": meta_emb,
        "note": "E2E = embedding time + classifier time for one request. OpenAI times include the "
                "network round trip to the API. Max-Prob-Diff compares each probability with the one "
                "the training run stored for the same test row.",
    }
    (args.out / "end_to_end_latency_meta.json").write_text(json.dumps(meta, indent=2))

    show = summary[["Pipeline", "Embedding", "Classifier", "Embed-P50(ms)", "Classify-P50(ms)", "E2E-P50(ms)",
                    "E2E-P95(ms)", "E2E-P99(ms)", "Max-Prob-Diff"]]
    print("\n" + show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nSaved to {args.out / 'end_to_end_latency.csv'} (+ _requests.csv, _meta.json)")
    worst = summary["Max-Prob-Diff"].max()
    if worst > 0.01:
        print(f"WARNING: a live probability differs from the stored one by {worst:.3f}; "
              "check that the embedding model and text match the training run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
