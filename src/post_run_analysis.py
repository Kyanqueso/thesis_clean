"""Write the ECE, disk-usage and train/test duplicate tables from the finished runs, without changing them."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from pipeline_paths import DATA_DIR, DATASETS, MODES, RUNS_DIR  # noqa: E402

N_BINS = 10   # ECE: equal-width bins, confidence = max(p, 1 - p) at the 0.5 cutoff (Chapter 3)
OUT_DIR = RUNS_DIR / "analysis"   # ece.csv, disk_usage.csv, duplicate_scan.csv


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    """Weighted gap between confidence and accuracy over equal-width confidence bins."""
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1.0 - p)
    correct = (pred == y).astype(float)
    # bin m holds confidences in ((m-1)/M, m/M]
    idx = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def ece_table() -> pd.DataFrame:
    rows = []
    for pipe in DATASETS:
        for mode in MODES:
            path = RUNS_DIR / pipe / mode / "results" / "predictions.npz"
            if not path.is_file():
                continue
            z = np.load(path)
            y = z["y_true"].astype(int)
            for key in z.files:
                if key in ("y_true", "test_idx"):
                    continue
                emb, _, clf = key.partition("__")
                p = z[key].astype(np.float64)
                rows.append({"Pipeline": pipe, "Mode": mode, "Embeddings": emb.upper(), "Classifier": clf,
                             "ECE": ece(y, p), "Accuracy@0.5": float(((p >= 0.5) == y).mean()),
                             "Bins": N_BINS, "Test-Samples": len(y)})
    return pd.DataFrame(rows)


def disk_table() -> pd.DataFrame:
    rows = []
    for pipe in DATASETS:
        for mode in MODES:
            run = RUNS_DIR / pipe / mode
            for f in sorted((run / "models").glob("*.joblib")):
                emb, _, clf = f.stem.partition("_")
                rows.append({"Pipeline": pipe, "Mode": mode, "Kind": "model", "Embeddings": emb.upper(),
                             "Classifier": clf, "File": f.name, "MB": f.stat().st_size / 1e6})
            for f in sorted((run / "embeddings").glob("*_prompt.npy")):
                rows.append({"Pipeline": pipe, "Mode": mode, "Kind": "embeddings",
                             "Embeddings": f.stem.replace("_prompt", "").upper(), "Classifier": "",
                             "File": f.name, "MB": f.stat().st_size / 1e6})
    return pd.DataFrame(rows)


def duplicate_table() -> pd.DataFrame:
    """Test rows whose text also appears among the training rows (fit + calibration)."""
    rows = []
    for pipe, name in DATASETS.items():
        splits = {}
        for mode in MODES:
            path = RUNS_DIR / pipe / mode / "results" / "predictions.npz"
            if path.is_file():
                splits[mode] = np.load(path)["test_idx"].astype(np.int64)
        if not splits:
            continue
        test_idx = next(iter(splits.values()))
        same_split = all(np.array_equal(np.sort(t), np.sort(test_idx)) for t in splits.values())

        df = pd.read_json(DATA_DIR / name, lines=True)
        is_test = np.zeros(len(df), dtype=bool)
        is_test[test_idx] = True
        intent = df["user_intent"].astype(str)
        context = df["context"].astype(str)
        fields = {"intent + context": intent + "\x00" + context, "context": context, "user intent": intent}
        if "original_context" in df.columns:   # Pipeline 3: the document before injection
            fields["original document"] = df["original_context"].astype(str)
        for field, text in fields.items():
            train_texts = set(text[~is_test])
            shared = text[is_test].isin(train_texts)
            rows.append({"Pipeline": pipe, "Field": field, "Test-Rows": int(is_test.sum()),
                         "Test-Rows-Shared-With-Train": int(shared.sum()),
                         "Share": float(shared.mean()),
                         "Distinct-Test-Texts": int(text[is_test].nunique()),
                         "Same-Split-All-Modes": same_split})
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in (("ece", ece_table), ("disk_usage", disk_table), ("duplicate_scan", duplicate_table)):
        table = build()
        if table.empty:
            print(f"{name}: nothing to report (no finished runs under {RUNS_DIR})")
            continue
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} rows -> {OUT_DIR / (name + '.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
