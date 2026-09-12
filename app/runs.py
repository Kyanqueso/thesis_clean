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
    return {"size_mb": round(st.st_size / 1e6, 2), "size": st.st_size,
            "mtime": int(st.st_mtime)}


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


def _finite(value):
    """NaN/inf -> None. Starlette's JSONResponse serialises with allow_nan=False,
    so a single NaN raises instead of rendering — and NaN is routine here: runs
    written before the metrics expansion have fewer columns, so concatenating
    them with newer runs fills the gaps."""
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _json_rows(df: pd.DataFrame) -> list[dict]:
    return [{k: _finite(v) for k, v in rec.items()} for rec in df.to_dict("records")]


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
    return _json_rows(pd.concat(frames, ignore_index=True))


# ---------------- master table ----------------
# One row per (pipeline, embedding, classifier): the reference mode's metrics,
# what the ablations cost, and whether that cost is significant. Everything a
# results screenshot needs, across all pipelines, in one grid.

MASTER_QUALITY = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
MASTER_EXTRA = ["FPR", "FNR", "TN", "FP", "FN", "TP", "Test-Samples",
                "Latency-P50(ms)", "Latency-P99(ms)", "Train-Time(s)",
                "Train-PeakRSS(MB)", "Train-PeakVRAM(MB)",
                "Inference-Time-per-Sample(ms)"]


def _ablation_significance(pipeline: str) -> dict:
    """(EMB, Clf, mode_a, mode_b) -> McNemar result, from the cross-ablation CSV.

    Written by src/train_evaluate_visualize.py into <runs>/{pipeline}/. Stored
    under both orderings: McNemar is symmetric, only the sign of the accuracy
    difference flips, and the table asks for whichever direction it renders.
    """
    path = RUNS_DIR / pipeline / "ablation_mcnemar.csv"
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, OSError, ValueError):
        return {}
    out = {}
    for r in _json_rows(df):
        emb = str(r.get("Embeddings", "")).upper()
        clf, a, b = str(r.get("Classifier", "")), r.get("Mode-A"), r.get("Mode-B")
        val = {"p_value": r.get("p_value"), "p_holm": r.get("p_holm"),
               "significant": bool(r.get("significant")),
               "b": r.get("b"), "c": r.get("c"), "method": r.get("method")}
        out[(emb, clf, a, b)] = val
        out[(emb, clf, b, a)] = val
    return out


def master_table(reference: str = "normal") -> dict:
    """Flat rows joining reference-mode metrics + ablation degradation + McNemar.

    Degradation is computed here from the per-run CSVs rather than read from
    ablation_degradation.csv, so the table still works on runs produced before
    the metrics expansion — those carry Accuracy/ROC-AUC/PR-AUC for all three
    modes, which is enough. McNemar cannot be reconstructed that way (it needs
    per-sample predictions), so it stays empty until the runs are redone.
    """
    if reference not in MODES:
        reference = "normal"
    records = all_results()
    variants = [m for m in MODES if m != reference]
    if not records:
        return {"rows": [], "reference": reference, "variants": variants,
                "metrics": MASTER_QUALITY, "notes": ["No results on disk yet."]}

    by = {(r["Pipeline"], r["Embeddings"], r["Classifier"], r["Mode"]): r for r in records}
    sig = {p: _ablation_significance(p) for p in PIPELINES}

    configs = sorted({(r["Pipeline"], r["Embeddings"], r["Classifier"]) for r in records},
                     key=lambda c: (list(PIPELINES).index(c[0]) if c[0] in PIPELINES else 99,
                                    c[1], c[2]))
    rows, missing_ref = [], 0
    for pipeline, emb, clf in configs:
        ref = by.get((pipeline, emb, clf, reference))
        if ref is None:
            missing_ref += 1
            continue
        row = {"Pipeline": pipeline, "Embeddings": emb, "Classifier": clf}
        for key in MASTER_QUALITY + MASTER_EXTRA:
            row[key] = _finite(ref.get(key))

        flags = []
        for mode in variants:
            var = by.get((pipeline, emb, clf, mode))
            for metric in MASTER_QUALITY:
                ref_v, var_v = (ref.get(metric), var.get(metric)) if var else (None, None)
                row[f"Deg:{mode}:{metric}"] = (
                    None if ref_v is None or var_v is None else float(ref_v) - float(var_v))
                row[f"Var:{mode}:{metric}"] = _finite(var_v) if var else None
            hit = sig.get(pipeline, {}).get((emb.upper(), clf, reference, mode))
            row[f"Sig:{mode}:p_holm"] = hit["p_holm"] if hit else None
            row[f"Sig:{mode}:significant"] = hit["significant"] if hit else None
            row[f"Sig:{mode}:b"] = hit["b"] if hit else None
            row[f"Sig:{mode}:c"] = hit["c"] if hit else None

        # Intent-only is the leakage canary: under a same-document design the
        # intent carries no label information, so it must sit at chance. If the
        # ablation costs nothing (or pays), the classes are separable without the
        # injection and the headline number is not injection detection.
        intent_auc = row.get("Var:user_intent_only:ROC-AUC")
        intent_deg = row.get("Deg:user_intent_only:ROC-AUC")
        if intent_auc is not None and intent_auc >= 0.99:
            flags.append("shortcut")
        elif intent_deg is not None and intent_deg <= 0:
            flags.append("shortcut")
        row["Flags"] = flags
        rows.append(row)

    notes = []
    if not any(sig.values()):
        notes.append("McNemar columns are empty: no <runs>/<pipeline>/ablation_mcnemar.csv. "
                     "It needs per-sample predictions.npz from every mode — re-run the "
                     "pipelines with the current src/train_evaluate_visualize.py.")
    if rows and rows[0].get("Precision") is None:
        notes.append("Precision/Recall/confusion/latency columns are empty: these runs predate "
                     "the metrics expansion. Degradation still works (it only needs "
                     "Accuracy/ROC-AUC/PR-AUC, which the old schema has).")
    if missing_ref:
        notes.append(f"{missing_ref} configuration(s) have no '{reference}' run to anchor on "
                     f"and are omitted; switch the reference mode to see them.")
    return {"rows": rows, "reference": reference, "variants": variants,
            "metrics": MASTER_QUALITY, "notes": notes}


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
        "fix": ("re-run that pipeline/mode from the Run tab (save models is on by "
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


# What /api/file/inspect can actually show. Anything else answers with a size
# line only, which is not a preview, so the UI offers no eye for it.
PREVIEW_EXT = frozenset({".csv", ".json", ".jsonl", ".log", ".txt", ".md",
                         ".html", ".npy", ".npz", ".png"})


def _tree_entry(rel: str, kind: str, mode: str | None = None,
                label: str | None = None) -> dict:
    target = paths.resolve(rel)
    info = _file_info(target) if target else None
    return {"path": rel, "kind": kind,  # expected | optional | unexpected
            "label": label or rel.rsplit("/", 1)[-1],
            "preview": bool(info) and Path(rel).suffix.lower() in PREVIEW_EXT,
            "mode": mode,  # the UI prefixes the label with this when set
            "status": "present" if info else "missing",
            "size": info["size"] if info else None,
            "mtime": info["mtime"] if info else None}


def _listdir_extra(dirpath: Path, base: str, known: set[str], kind: str,
                   mode: str | None = None) -> list[dict]:
    out = []
    if dirpath.is_dir():
        for p in sorted(dirpath.iterdir()):
            if p.is_file() and p.name not in known:
                out.append(_tree_entry(f"{base}/{p.name}", kind, mode))
    return out


def _node(name: str, entries: list[dict], children: list[dict] | None = None) -> dict:
    """One tree node with its counts rolled up from whatever sits under it.

    n_expected/n_present describe the schema (what a complete run looks like);
    n_files is everything actually on disk, which is all a figures/models node
    can report since nothing there is enumerable up front.
    """
    children = children or []
    exp = [e for e in entries if e["kind"] == "expected"]
    return {
        "name": name,
        "entries": entries,
        "children": children,
        "n_expected": len(exp) + sum(c["n_expected"] for c in children),
        "n_present": sum(e["status"] == "present" for e in exp)
                     + sum(c["n_present"] for c in children),
        "n_files": sum(e["status"] == "present" for e in entries)
                   + sum(c["n_files"] for c in children),
    }


def file_tree() -> dict:
    """Artifacts grouped by kind, then by pipeline.

    Two levels only: the leaf table under a pipeline holds every ablation, so
    any file is two clicks away.
    """
    ds_known = {cfg["data"] for cfg in PIPELINES.values()}
    ds = [_tree_entry(f"{paths.DATA_PREFIX}/{cfg['data']}", "expected")
          for cfg in PIPELINES.values()]
    ds += _listdir_extra(DATA_DIR, paths.DATA_PREFIX, ds_known, "unexpected")

    emb_known = {f"{m}_prompt.npy" for m in EMB_MODELS}
    kinds: dict[str, list[dict]] = {k: [] for k in ("Embeddings", "Results", "Figures", "Models")}
    for pipeline, cfg in PIPELINES.items():
        bucket: dict[str, list[dict]] = {k: [] for k in kinds}
        for mode in MODES:
            base = f"{paths.RUNS_PREFIX}/{pipeline}/{mode}"
            d = run_dir(pipeline, mode)
            bucket["Embeddings"] += [
                _tree_entry(f"{base}/embeddings/{m}_prompt.npy", "expected", mode)
                for m in EMB_MODELS]
            bucket["Embeddings"] += _listdir_extra(
                d / "embeddings", f"{base}/embeddings", emb_known, "unexpected", mode)
            bucket["Results"] += [_tree_entry(f"{base}/results/{n}", "expected", mode)
                                  for n in EXPECTED_RESULTS]
            bucket["Results"] += _listdir_extra(
                d / "results", f"{base}/results", set(EXPECTED_RESULTS), "unexpected", mode)
            bucket["Figures"] += _listdir_extra(
                d / "figures", f"{base}/figures", set(), "optional", mode)
            bucket["Models"] += _listdir_extra(
                d / "models", f"{base}/models", set(), "optional", mode)
        short = cfg["label"].split(" (")[0]  # drop the parenthetical; the tree is narrow
        for kind, entries in bucket.items():
            kinds[kind].append(_node(short, entries))

    groups = [_node("Input datasets", ds)]
    groups += [_node(kind, [], children) for kind, children in kinds.items()]
    groups.append(_node("Other", [], _other_children()))
    return {"groups": groups}


def _other_children() -> list[dict]:
    """Everything that is neither dataset nor part of a run tree: the sweep
    aggregate, smoke output, and files left in the repo root by scripts run
    with their default dirs."""
    out = [_node("Summary", [_tree_entry(paths.summary_rel(n), "expected")
                             for n in paths.SUMMARY_FILES])]

    # smoke runs (--limit) land in their own subtree so they cannot overwrite a
    # delivered run; list whatever is there so the output is still reachable
    smoke_root = paths.RUNS_WRITE_DIR / paths.SMOKE_SUBDIR
    smoke_prefix = f"{paths.WRITE_PREFIX}/{paths.SMOKE_SUBDIR}"
    smoke = []
    if smoke_root.is_dir():
        for sub_dir in sorted(d for d in smoke_root.rglob("*") if d.is_dir()):
            rel = sub_dir.relative_to(smoke_root).as_posix()
            for e in _listdir_extra(sub_dir, f"{smoke_prefix}/{rel}", set(), "optional"):
                e["label"] = f"{rel}/{e['label']}"
                smoke.append(e)
    if smoke:
        out.append(_node("Smoke runs", smoke))

    legacy = []
    for sub in ("embeddings", "results", "figures"):
        for e in _listdir_extra(ROOT / sub, sub, set(), "optional"):
            e["label"] = e["path"]
            legacy.append(e)
    if legacy:
        out.append(_node("Legacy root output", legacy))
    return out


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
                "fix": "download or build it from the Run tab, then retry"}
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
