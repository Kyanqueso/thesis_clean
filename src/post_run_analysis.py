"""Write the ECE, disk-usage, train/test duplicate and cross-pipeline z-test tables from the finished runs,
without changing them."""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from pipeline_paths import DATA_DIR, DATASETS, MODES, RUNS_DIR  # noqa: E402

N_BINS = 10   # ECE: equal-width bins, confidence = max(p, 1 - p) at the 0.5 cutoff (Chapter 3)
OUT_DIR = RUNS_DIR / "analysis"   # ece.csv, disk_usage.csv, duplicate_scan.csv, two_proportion_z.csv

# two-proportion z-test: Pipelines 2 and 3 against Pipeline 1, normal mode, matched embedding x classifier
Z_REFERENCE = "pipeline1_alamsabi"
Z_MODE = "normal"
Z_ALPHA = 0.05   # family-wise: applied to the Holm-adjusted p-values, so it holds across all the tests at once
Z_CRIT = norm.ppf(1 - Z_ALPHA / 2)   # 1.96, for the 95% interval around each accuracy gap


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


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values: the smallest p is multiplied by m, the next by m - 1, and so on,
    each kept at least as large as the one before it and capped at 1."""
    order = np.argsort(p_values)
    adjusted, running, m = [0.0] * len(p_values), 0.0, len(p_values)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def two_proportion_z(x1: int, n1: int, x2: int, n2: int) -> dict:
    """Two-sided two-proportion z-test of accuracy x1/n1 against x2/n2, plus a 95% interval for the gap."""
    p1, p2 = x1 / n1, x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    # the test assumes both accuracies are equal, so its standard error uses the pooled proportion
    se_test = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se_test if se_test > 0 else 0.0   # both perfect (or both zero): no gap to test
    # the interval describes the gap as observed, so it uses each accuracy's own standard error
    se_gap = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    return {"z": z, "p_value": float(2 * norm.sf(abs(z))), "Diff": diff,
            "CI95-Low": diff - Z_CRIT * se_gap, "CI95-High": diff + Z_CRIT * se_gap}


def z_test_table() -> pd.DataFrame:
    """Accuracy of each Pipeline 2 and 3 model against the same embedding x classifier in Pipeline 1.

    Independence: the z-test treats the two test sets as independent samples. Each pipeline's test set comes
    from its own dataset file and split, but the datasets share sources: Pipeline 2's malicious rows are drawn
    from Pipeline 1's BIPIA rows, so a few test rows can appear in both test sets, and Pipeline 3 uses BIPIA
    attack instructions too. Any overlap makes the samples only approximately independent, so the p-values
    are approximate. The test compares accuracy on different data: a gap means the pipelines' datasets differ
    in how hard they are to classify, not that one classifier is better than another.
    """
    def load(pipeline: str) -> pd.DataFrame | None:
        path = RUNS_DIR / pipeline / Z_MODE / "results" / "full_evaluation_results.csv"
        if not path.is_file():
            return None
        df = pd.read_csv(path)
        df["x"] = df["TP"] + df["TN"]                           # correct predictions at the 0.5 threshold
        df["n"] = df[["TN", "FP", "FN", "TP"]].sum(axis=1)       # test samples
        return df[["Embeddings", "Classifier", "x", "n"]]

    ref = load(Z_REFERENCE)
    if ref is None:
        return pd.DataFrame()
    rows = []
    for pipe in DATASETS:
        other = load(pipe) if pipe != Z_REFERENCE else None
        if other is None:
            continue
        pairs = other.merge(ref, on=["Embeddings", "Classifier"], suffixes=("", "_ref"))
        for r in pairs.itertuples(index=False):
            res = two_proportion_z(int(r.x), int(r.n), int(r.x_ref), int(r.n_ref))
            rows.append({
                "Comparison": f"{pipe.split('_')[0]} vs {Z_REFERENCE.split('_')[0]}",
                "Embeddings": r.Embeddings, "Classifier": r.Classifier,
                "Acc-Other": r.x / r.n, "Acc-P1": r.x_ref / r.n_ref,
                "Diff(pp)": 100 * res["Diff"],
                "CI95-Low(pp)": 100 * res["CI95-Low"], "CI95-High(pp)": 100 * res["CI95-High"],
                "z": res["z"], "p_value": res["p_value"],
                "n-Other": int(r.n), "n-P1": int(r.n_ref),
            })
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # one family: every comparison in the table, so the chance of any false "significant" stays at Z_ALPHA
    table["p_holm"] = holm(table["p_value"].tolist())
    table["significant"] = table["p_holm"] <= Z_ALPHA
    return table.sort_values(["Comparison", "Embeddings", "Classifier"]).reset_index(drop=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, build in (("ece", ece_table), ("disk_usage", disk_table), ("duplicate_scan", duplicate_table),
                        ("two_proportion_z", z_test_table)):
        table = build()
        if table.empty:
            print(f"{name}: nothing to report (no finished runs under {RUNS_DIR})")
            continue
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)
        print(f"{name}: {len(table)} rows -> {OUT_DIR / (name + '.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
