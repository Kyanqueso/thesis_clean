"""Read side of the dashboard: discovery and parsing of the runs tree.

Layout produced by src/run_all_pipelines.py:
  <runs>/{pipeline}/{mode}/{embeddings,results,figures,models}/

<runs> and the dataset dir are located per machine by app.paths (runs/ + data/
here, runs_from_vastai/ + data_from_vastai/ on the main machine).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, auc

from app import paths
from app.paths import DATA_DIR, ROOT, RUNS_DIR  # noqa: F401  (re-exported for detect/jobs)

PIPELINES = {
    "pipeline1_alamsabi": {
        "label": "P1 · Alamsabi (BIPIA malicious + GPT benign)",
        "data": "final_training_dataset.jsonl",
    },
    "pipeline2_organic_bipia": {
        "label": "P2 · Organic benign + BIPIA malicious",
        "data": "organic_bipia_dataset.jsonl",
    },
    "pipeline3_organic_injected": {
        "label": "P3 · Organic, self-injected (doc-id guarded)",
        "data": "organic_injected_dataset.jsonl",
    },
}
MODES = ["normal", "user_intent_only", "context_only"]
EMB_MODELS = ["openai", "qwen3", "minilm"]

_CURVE_CACHE: dict[tuple, dict] = {}


def run_dir(pipeline: str, mode: str) -> Path:
    return RUNS_DIR / pipeline / mode


def data_path(pipeline: str) -> Path:
    return DATA_DIR / PIPELINES[pipeline]["data"]


def _file_info(p: Path) -> dict | None:
    if not p.is_file():
        return None
    st = p.stat()
    return {"size_mb": round(st.st_size / 1e6, 2), "mtime": int(st.st_mtime)}


def scan_state() -> dict:
    """Full artifact matrix: datasets + per-run embeddings/results/figures/models."""
    datasets = {}
    for key, cfg in PIPELINES.items():
        datasets[key] = {"label": cfg["label"], "file": cfg["data"],
                         "info": _file_info(data_path(key))}

    runs = []
    for pipeline in PIPELINES:
        for mode in MODES:
            d = run_dir(pipeline, mode)
            embeds = {m: (d / "embeddings" / f"{m}_prompt.npy").is_file() for m in EMB_MODELS}
            results_csv = d / "results" / "full_evaluation_results.csv"
            meta_path = d / "results" / "run_meta.json"
            meta = None
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    meta = None
            runs.append({
                "pipeline": pipeline,
                "mode": mode,
                "embeddings": embeds,
                "has_results": results_csv.is_file(),
                "has_predictions": (d / "results" / "predictions.npz").is_file(),
                "results_info": _file_info(results_csv),
                "figures": sorted(p.name for p in (d / "figures").glob("*.png")) if (d / "figures").is_dir() else [],
                "models": sorted(p.stem for p in (d / "models").glob("*.joblib")) if (d / "models").is_dir() else [],
                "meta": meta,
            })

    return {
        "datasets": datasets,
        "pipelines": {k: v["label"] for k, v in PIPELINES.items()},
        "modes": MODES,
        "emb_models": EMB_MODELS,
        "runs": runs,
        "has_sweep_summary": paths.summary_path("all_pipelines_summary.csv") is not None,
        "paths": paths.describe(),
        # delivered artifacts carry results and figures but no models or
        # per-sample probabilities; the UI explains what that switches off
        "capabilities": {
            "n_runs_with_results": sum(r["has_results"] for r in runs),
            "n_runs_with_predictions": sum(r["has_predictions"] for r in runs),
            "n_runs_with_models": sum(bool(r["models"]) for r in runs),
        },
    }


def all_results() -> list[dict]:
    """Every run's full_evaluation_results.csv, tagged with Pipeline/Mode."""
    frames = []
    for pipeline in PIPELINES:
        for mode in MODES:
            csv_path = run_dir(pipeline, mode) / "results" / "full_evaluation_results.csv"
            if not csv_path.is_file():
                continue
            try:
                df = pd.read_csv(csv_path)
            except (pd.errors.ParserError, OSError):
                continue
            df.insert(0, "Pipeline", pipeline)
            df.insert(1, "Mode", mode)
            frames.append(df)
    if not frames:
        return []
    return pd.concat(frames, ignore_index=True).to_dict("records")


def _downsample(x: np.ndarray, y: np.ndarray, n: int = 250) -> tuple[list, list]:
    if len(x) <= n:
        return x.tolist(), y.tolist()
    idx = np.unique(np.linspace(0, len(x) - 1, n).astype(int))
    return x[idx].tolist(), y[idx].tolist()


def _unavailable(what: str, pipeline: str, mode: str | None = None) -> dict:
    """Uniform 'this tree cannot answer that' payload, with the way out."""
    where = f"{pipeline}/{mode}" if mode else pipeline
    return {
        "available": False,
        "missing": what,
        "reason": f"no {what} in {where} — delivered runs ship results and figures only",
        "fix": ("re-run that pipeline/mode from the Runs tab (save models is on by "
                "default) to produce per-sample probabilities and saved classifiers"),
    }


def curves(pipeline: str, mode: str) -> dict:
    """ROC/PR points + score histograms for every emb x clf combo of one run,
    computed from the per-sample probabilities in predictions.npz."""
    npz_path = run_dir(pipeline, mode) / "results" / "predictions.npz"
    if not npz_path.is_file():
        return _unavailable("predictions.npz", pipeline, mode)

    cache_key = (str(npz_path), npz_path.stat().st_mtime)
    if cache_key in _CURVE_CACHE:
        return _CURVE_CACHE[cache_key]

    data = np.load(npz_path)
    y_true = data["y_true"]
    combos = {}
    bins = np.linspace(0, 1, 201)  # 200 bins -> slider step of 0.005 stays exact
    for key in data.files:
        if key in ("y_true", "test_idx"):
            continue
        y_prob = data[key]
        emb, clf = key.split("__", 1)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        fpr_d, tpr_d = _downsample(fpr, tpr)
        rec_d, prec_d = _downsample(rec[::-1], prec[::-1])
        combos[key] = {
            "emb": emb, "clf": clf,
            "roc": {"fpr": fpr_d, "tpr": tpr_d, "auc": float(roc_auc_score(y_true, y_prob))},
            "pr": {"recall": rec_d, "precision": prec_d, "auc": float(auc(rec[::-1], prec[::-1]))},
            "hist_benign": np.histogram(y_prob[y_true == 0], bins=bins)[0].tolist(),
            "hist_malicious": np.histogram(y_prob[y_true == 1], bins=bins)[0].tolist(),
        }

    out = {"available": True, "pipeline": pipeline, "mode": mode,
           "n_test": int(len(y_true)), "n_pos": int(y_true.sum()), "combos": combos}
    _CURVE_CACHE.clear()  # keep at most one run cached in memory
    _CURVE_CACHE[cache_key] = out
    return out


def sample_rows(pipeline: str, n: int = 1, seed: int | None = None) -> list[dict]:
    """Reservoir-sample n rows from a pipeline's jsonl dataset."""
    import random
    path = data_path(pipeline)
    if not path.is_file():
        return []
    rng = random.Random(seed)
    reservoir: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(reservoir) < n:
                reservoir.append(line)
            else:
                j = rng.randint(0, i)
                if j < n:
                    reservoir[j] = line
    rows = []
    for line in reservoir:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


# ---------------- file explorer (expected-vs-actual manifest) ----------------

EXPECTED_RESULTS = ["full_evaluation_results.csv", "full_evaluation_results.html",
                    "evaluation_summary.json", "predictions.npz", "run_meta.json"]


def _tree_entry(rel: str, kind: str) -> dict:
    target = paths.resolve(rel)
    info = _file_info(target) if target else None
    return {"path": rel, "kind": kind,  # expected | optional | unexpected
            "status": "present" if info else "missing",
            "size_mb": info["size_mb"] if info else None,
            "mtime": info["mtime"] if info else None}


def _listdir_extra(dirpath: Path, base: str, known: set[str], kind: str) -> list[dict]:
    out = []
    if dirpath.is_dir():
        for p in sorted(dirpath.iterdir()):
            if p.is_file() and p.name not in known:
                out.append(_tree_entry(f"{base}/{p.name}", kind))
    return out


def file_tree() -> dict:
    """Every artifact the pipeline schema expects (present or missing) plus any
    extra files found on disk in those directories."""
    sections = []

    def section(name, entries):
        exp = [e for e in entries if e["kind"] == "expected"]
        sections.append({"name": name, "entries": entries,
                         "n_expected": len(exp),
                         "n_present": sum(e["status"] == "present" for e in exp)})

    ds_known = {cfg["data"] for cfg in PIPELINES.values()}
    ds_entries = [_tree_entry(f"{paths.DATA_PREFIX}/{cfg['data']}", "expected") for cfg in PIPELINES.values()]
    ds_entries += _listdir_extra(DATA_DIR, paths.DATA_PREFIX, ds_known, "unexpected")
    section(f"{paths.DATA_PREFIX} — input datasets", ds_entries)

    emb_known = {f"{m}_prompt.npy" for m in EMB_MODELS}
    for pipeline in PIPELINES:
        for mode in MODES:
            base = f"{paths.RUNS_PREFIX}/{pipeline}/{mode}"
            d = run_dir(pipeline, mode)
            entries = [_tree_entry(f"{base}/embeddings/{m}_prompt.npy", "expected") for m in EMB_MODELS]
            entries += [_tree_entry(f"{base}/results/{name}", "expected") for name in EXPECTED_RESULTS]
            entries += _listdir_extra(d / "embeddings", f"{base}/embeddings", emb_known, "unexpected")
            entries += _listdir_extra(d / "results", f"{base}/results", set(EXPECTED_RESULTS), "unexpected")
            for sub in ("figures", "models"):
                entries += _listdir_extra(d / sub, f"{base}/{sub}", set(), "optional")
            section(f"{pipeline} / {mode}", entries)

    section(f"{paths.RUNS_PREFIX} — sweep aggregate",
            [_tree_entry(paths.summary_rel(name), "expected") for name in paths.SUMMARY_FILES])

    # smoke runs (--limit) land in their own subtree so they cannot overwrite a
    # delivered run; list whatever is there so the output is still reachable
    smoke_root = paths.RUNS_WRITE_DIR / paths.SMOKE_SUBDIR
    smoke_prefix = f"{paths.WRITE_PREFIX}/{paths.SMOKE_SUBDIR}"
    if smoke_root.is_dir():
        smoke = []
        for sub_dir in sorted(d for d in smoke_root.rglob("*") if d.is_dir()):
            rel = sub_dir.relative_to(smoke_root).as_posix()
            smoke += _listdir_extra(sub_dir, f"{smoke_prefix}/{rel}", set(), "optional")
        if smoke:
            section("smoke runs (--limit output)", smoke)

    # scripts run with their default dirs write to the repo root instead of runs/
    legacy = []
    for sub in ("embeddings", "results", "figures"):
        legacy += _listdir_extra(ROOT / sub, sub, set(), "optional")
    if legacy:
        section("legacy root outputs (scripts run with default dirs)", legacy)

    return {"sections": sections}


# ---------------- per-group breakdown (needs predictions.npz + dataset) ----------------

BREAKDOWN_FIELDS = ["category", "attack_category", "injection_position", "source"]
_BREAKDOWN_CACHE: dict = {}


def _dataset_meta(pipeline: str) -> pd.DataFrame | None:
    """Metadata columns of a pipeline dataset, cached by (path, mtime)."""
    path = data_path(pipeline)
    if not path.is_file():
        return None
    key = (str(path), path.stat().st_mtime)
    if _BREAKDOWN_CACHE.get("key") == key:
        return _BREAKDOWN_CACHE["df"]
    df = pd.read_json(path, lines=True)
    keep = [c for c in BREAKDOWN_FIELDS + ["label"] if c in df.columns]
    df = df[keep]
    _BREAKDOWN_CACHE.clear()
    _BREAKDOWN_CACHE["key"] = key
    _BREAKDOWN_CACHE["df"] = df
    return df


def breakdown_fields(pipeline: str) -> dict:
    df = _dataset_meta(pipeline)
    if df is None:
        name = PIPELINES[pipeline]["data"]
        return {"available": False, "missing": name,
                "reason": f"{name} is not in {paths.DATA_PREFIX}/ — the breakdown reads that dataset's metadata columns",
                "fix": "download or build it from the Runs tab, then retry"}
    return {"available": True, "fields": [c for c in BREAKDOWN_FIELDS if c in df.columns]}


def breakdown(pipeline: str, mode: str, combo_key: str, by: str) -> dict:
    npz_path = run_dir(pipeline, mode) / "results" / "predictions.npz"
    if not npz_path.is_file():
        return _unavailable("predictions.npz", pipeline, mode)
    df = _dataset_meta(pipeline)
    if df is None:
        return breakdown_fields(pipeline)
    if by not in df.columns:
        return {"available": False, "missing": by,
                "reason": f"field '{by}' is not in this dataset",
                "fix": f"pick one of: {', '.join(c for c in BREAKDOWN_FIELDS if c in df.columns)}"}

    data = np.load(npz_path)
    if combo_key not in data.files:
        return {"available": False, "missing": combo_key,
                "reason": f"no stored predictions for {combo_key} in {pipeline}/{mode}",
                "fix": f"available combos: {', '.join(k for k in data.files if k not in ('y_true', 'test_idx'))}"}
    y_true = data["y_true"].astype(bool)
    test_idx = data["test_idx"]
    y_pred = data[combo_key] >= 0.5

    sub = df.iloc[test_idx].reset_index(drop=True)
    vals = sub[by]
    rows = []
    for value, mask in ((v, (vals == v).to_numpy()) for v in vals.dropna().unique()):
        n = int(mask.sum())
        mal = mask & y_true
        row = {
            "value": str(value),
            "n": n,
            "n_malicious": int(mal.sum()),
            "accuracy": float((y_pred[mask] == y_true[mask]).mean()),
            "detection_rate": float(y_pred[mal].mean()) if mal.any() else None,
            "false_positive_rate": float(y_pred[mask & ~y_true].mean()) if (mask & ~y_true).any() else None,
        }
        rows.append(row)
    rows.sort(key=lambda r: -r["n"])
    n_null = int(vals.isna().sum())
    return {"available": True, "by": by, "combo": combo_key, "rows": rows,
            "note": f"{n_null} test rows have no '{by}' value (benign rows for attack fields)" if n_null else ""}
