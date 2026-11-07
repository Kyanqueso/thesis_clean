"""Read side of the dashboard: what src/train_evaluate_visualize.py left in <runs>/{pipeline}/{mode}/."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from app import paths
from pipeline_paths import DATASETS, MODES  # noqa: F401  (MODES is re-exported)

# read paths.RUNS_DIR / paths.DATA_DIR through the module, so tests can point them elsewhere
LABELS = {
    "pipeline1_alamsabi": "P1 · Alamsabi (BIPIA malicious + GPT benign)",
    "pipeline2_organic_bipia": "P2 · Organic benign + BIPIA malicious",
    "pipeline3_organic_injected": "P3 · Organic, self-injected (doc-id guarded)",
}
PIPELINES = {key: {"label": LABELS[key], "data": name} for key, name in DATASETS.items()}
EMB_MODELS = ["openai", "qwen3", "minilm"]


def run_dir(pipeline: str, mode: str) -> Path:
    return paths.RUNS_DIR / pipeline / mode


def data_path(pipeline: str) -> Path:
    return paths.DATA_DIR / PIPELINES[pipeline]["data"]


def _file_info(p: Path) -> dict | None:
    if not p.is_file():
        return None
    st = p.stat()
    return {"size_mb": round(st.st_size / 1e6, 2), "size": st.st_size, "mtime": int(st.st_mtime)}


def scan_state() -> dict:
    """Datasets and per-run embeddings/results/figures/models, read fresh on every poll."""
    datasets = {key: {"label": cfg["label"], "file": cfg["data"], "info": _file_info(data_path(key))}
                for key, cfg in PIPELINES.items()}

    runs = []
    for pipeline in PIPELINES:
        for mode in MODES:
            d = run_dir(pipeline, mode)
            results_csv = d / "results" / "full_evaluation_results.csv"
            meta = None
            try:
                meta = json.loads((d / "results" / "run_meta.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
            runs.append({
                "pipeline": pipeline,
                "mode": mode,
                "embeddings": {m: (d / "embeddings" / f"{m}_prompt.npy").is_file() for m in EMB_MODELS},
                "has_results": results_csv.is_file(),
                "has_predictions": (d / "results" / "predictions.npz").is_file(),
                "results_info": _file_info(results_csv),
                "figures": sorted(p.name for p in (d / "figures").glob("*.png")),
                "models": sorted(p.stem for p in (d / "models").glob("*.joblib")),
                "meta": meta,
            })

    return {
        "datasets": datasets,
        "pipelines": {k: v["label"] for k, v in PIPELINES.items()},
        "modes": MODES,
        "emb_models": EMB_MODELS,
        "runs": runs,
        "paths": paths.describe(),
        "capabilities": {
            "n_runs_with_results": sum(r["has_results"] for r in runs),
            "n_runs_with_predictions": sum(r["has_predictions"] for r in runs),
            "n_runs_with_models": sum(bool(r["models"]) for r in runs),
        },
    }


def _finite(value):
    """NaN/inf -> None (JSONResponse refuses NaN), numpy scalars -> Python ones."""
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _json_rows(df: pd.DataFrame) -> list[dict]:
    return [{k: _finite(v) for k, v in rec.items()} for rec in df.to_dict("records")]


def all_results() -> list[dict]:
    """Every run's full_evaluation_results.csv, with its operating_points.csv columns, tagged with Pipeline/Mode."""
    frames = []
    for pipeline in PIPELINES:
        for mode in MODES:
            results = run_dir(pipeline, mode) / "results"
            try:
                df = pd.read_csv(results / "full_evaluation_results.csv")
                if (results / "operating_points.csv").is_file():
                    df = df.merge(pd.read_csv(results / "operating_points.csv"),
                                  on=["Embeddings", "Classifier"], how="left")
            except (OSError, pd.errors.ParserError, KeyError):
                continue
            df.insert(0, "Pipeline", pipeline)
            df.insert(1, "Mode", mode)
            frames.append(df)
    return _json_rows(pd.concat(frames, ignore_index=True)) if frames else []


# ---------------- master table: one row per (pipeline, embedding, classifier) ----------------

MASTER_QUALITY = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]
MASTER_EXTRA = ["FPR", "TN", "FP", "FN", "TP", "Test-Samples",
                "TPR@1%FPR", "FPR@1%FPR", "TPR@0.1%FPR", "FPR@0.1%FPR",
                "Latency-P50(ms)", "Latency-P99(ms)", "Train-Time(s)", "Train-RAM(MB)",
                "Inference-Time-per-Sample(ms)"]


def _ablation_significance(pipeline: str) -> dict:
    """(EMB, Clf, mode_a, mode_b) -> McNemar result from <runs>/{pipeline}/ablation_mcnemar.csv, both orders."""
    try:
        df = pd.read_csv(paths.RUNS_DIR / pipeline / "ablation_mcnemar.csv")
    except (OSError, pd.errors.ParserError, ValueError):
        return {}
    out = {}
    for r in _json_rows(df):
        val = {"p_value": r.get("p_value"), "p_holm": r.get("p_holm"),
               "significant": bool(r.get("significant")),
               "b": r.get("b"), "c": r.get("c"), "method": r.get("method")}
        key = (str(r.get("Embeddings", "")).upper(), str(r.get("Classifier", "")))
        out[(*key, r.get("Mode-A"), r.get("Mode-B"))] = val
        out[(*key, r.get("Mode-B"), r.get("Mode-A"))] = val
    return out


def master_table(reference: str = "normal") -> dict:
    """Reference-mode metrics, ablation degradation (reference - variant) and McNemar, one row per config."""
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
                     key=lambda c: (list(PIPELINES).index(c[0]), c[1], c[2]))
    rows, missing_ref = [], 0
    for pipeline, emb, clf in configs:
        ref = by.get((pipeline, emb, clf, reference))
        if ref is None:
            missing_ref += 1
            continue
        row = {"Pipeline": pipeline, "Embeddings": emb, "Classifier": clf}
        for key in MASTER_QUALITY + MASTER_EXTRA:
            row[key] = _finite(ref.get(key))
        for mode in variants:
            var = by.get((pipeline, emb, clf, mode)) or {}
            for metric in MASTER_QUALITY:
                ref_v, var_v = ref.get(metric), var.get(metric)
                row[f"Deg:{mode}:{metric}"] = (
                    None if ref_v is None or var_v is None else float(ref_v) - float(var_v))
                row[f"Var:{mode}:{metric}"] = _finite(var_v)
            hit = sig[pipeline].get((emb.upper(), clf, reference, mode)) or {}
            for field in ("p_holm", "significant", "b", "c"):
                row[f"Sig:{mode}:{field}"] = hit.get(field)
        rows.append(row)

    notes = []
    if not any(sig.values()):
        notes.append("McNemar columns are empty: no <runs>/<pipeline>/ablation_mcnemar.csv yet. "
                     "It is written once two or more modes of a pipeline have been trained.")
    if missing_ref:
        notes.append(f"{missing_ref} configuration(s) have no '{reference}' run to anchor on "
                     f"and are omitted; switch the reference mode to see them.")
    return {"rows": rows, "reference": reference, "variants": variants,
            "metrics": MASTER_QUALITY, "notes": notes}


# ---------------- resources: what running the detector costs ----------------
# Two sources measure the same things. The normal-mode runs, as they go: the training run's cost columns
# and what generate_embeddings.py logged while embedding ({emb}_gpu.json, only when it embedded on a local
# CUDA GPU). And Run -> Measure resources (src/benchmark_resources.py -> BENCH_FILE), which re-measures
# them on their own. Per row the newer one is shown, with any blank filled from the other.

BENCH_FILE = "resource_benchmark.json"
EMB_LOCAL = {"minilm", "qwen3"}   # run on the GPU; openai runs on OpenAI's servers
CLF_COST = ("fit_s", "cpu_s", "ram_mb", "p50_ms", "p99_ms", "model_mb")
GPU_COST = ("vram_peak_mb", "gpu_util_mean", "gpu_util_max")


def _mb(p: Path) -> float | None:
    return round(p.stat().st_size / 1e6, 3) if p.is_file() else None


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _newer(run: dict, bench: dict, keys: tuple) -> tuple[dict, str | None, str | None]:
    """keys from the more recently measured source (run or benchmark), blanks filled from the other;
    both carry their time as "at". Returns (values, which source won, its time)."""
    srcs = [("run", run), ("benchmark", bench)]
    has = [(name, s) for name, s in srcs if any(_finite(s.get(k)) is not None for k in keys)]
    if not has:
        return {k: None for k in keys}, None, None
    has.sort(key=lambda ns: ns[1].get("at") or "", reverse=True)   # newest first
    out = {}
    for k in keys:
        out[k] = next((v for _, s in has if (v := _finite(s.get(k))) is not None), None)
    return out, has[0][0], has[0][1].get("at")


def resources() -> dict:
    """One row per pipeline x embedding x classifier (normal mode), embedding columns repeated per row."""
    bench = _read_json(paths.RUNS_DIR / BENCH_FILE)
    emb_b, clf_b = bench.get("embedding", {}), bench.get("classifier", {})
    rows, emb_info, metas, stamps = [], {}, {}, []
    for r in all_results():
        pipeline, emb, clf = r["Pipeline"], str(r["Embeddings"]).lower(), r["Classifier"]
        if r["Mode"] != "normal" or pipeline not in PIPELINES:
            continue
        d = run_dir(pipeline, "normal")
        if pipeline not in metas:
            metas[pipeline] = _read_json(d / "results" / "run_meta.json")
        if (pipeline, emb) not in emb_info:
            ef = d / "embeddings" / f"{emb}_prompt.npy"
            try:
                shape = np.load(ef, mmap_mode="r").shape[:2]
            except (OSError, ValueError):
                shape = (None, None)
            g = _read_json(d / "embeddings" / f"{emb}_gpu.json")
            e = emb_b.get(f"{pipeline}|{emb}") or {}
            gpu, gpu_src, gpu_at = _newer({**g, "at": g.get("measured_at")}, {**e, "at": e.get("measured_at")},
                                          GPU_COST)
            if gpu_at:
                stamps.append(gpu_at)
            emb_info[pipeline, emb] = (ef, shape, gpu, gpu_src,
                                       (g if gpu_src == "run" else e).get("device"))
        ef, (n_rows, dims), gpu, gpu_src, _ = emb_info[pipeline, emb]

        c = clf_b.get(f"{pipeline}|{emb}|{clf}") or {}
        run_cost = {"fit_s": r.get("Train-Time(s)"), "cpu_s": r.get("Train-CPU(s)"),
                    "ram_mb": r.get("Train-RAM(MB)"),   # older runs have no such column
                    "p50_ms": r.get("Latency-P50(ms)"), "p99_ms": r.get("Latency-P99(ms)"),
                    "model_mb": _mb(d / "models" / f"{emb}_{clf}.joblib"),
                    "at": metas[pipeline].get("finished_at")}
        # older benchmark entries have only peak_ram_mb, a whole-process peak: not the same thing, left out
        cost, clf_src, clf_at = _newer(run_cost, {**c, "at": c.get("measured_at")}, CLF_COST)
        if clf_at:
            stamps.append(clf_at)
        rows.append({
            "Pipeline": pipeline, "Embeddings": emb.upper(), "Classifier": clf,
            "emb_local": emb in EMB_LOCAL,
            "vram_mb": gpu["vram_peak_mb"], "gpu_mean": gpu["gpu_util_mean"], "gpu_max": gpu["gpu_util_max"],
            "gpu_src": gpu_src,
            "emb_disk_mb": _mb(ef), "emb_rows": n_rows, "emb_dims": dims,
            **cost, "clf_src": clf_src,
        })
    rows.sort(key=lambda r: (list(PIPELINES).index(r["Pipeline"]), r["Embeddings"], r["Classifier"]))

    env = next((m["environment"] for m in metas.values() if m.get("environment")), {})
    bm = bench.get("machine") or {}
    machine = {"cpu": bm.get("cpu") or env.get("cpu") or env.get("processor"),
               "ram_gb": env.get("total_ram_gb") or bm.get("ram_gb"),
               "gpu": next((dev for *_, dev in emb_info.values() if dev), None) or bm.get("gpu")}
    return {"rows": rows, "machine": machine, "measured_at": max(stamps) if stamps else None,
            "n_embedding": sum(v[3] is not None for v in emb_info.values()),
            "n_classifier": sum(r["clf_src"] is not None for r in rows),
            "n_benchmark": sum(r["clf_src"] == "benchmark" for r in rows), "bench_file": BENCH_FILE}


def sample_rows(pipeline: str, n: int = 1) -> list[dict]:
    """Reservoir-sample n rows from a pipeline's jsonl dataset."""
    path = data_path(pipeline)
    if not path.is_file():
        return []
    rng = random.Random()
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


# ---------------- Files tab: expected-vs-actual manifest ----------------

EXPECTED_RESULTS = ["full_evaluation_results.csv", "mcnemar_within_run.csv",
                    "operating_points.csv", "predictions.npz", "run_meta.json"]

# what /api/file/inspect can render; anything else gets no preview eye
PREVIEW_EXT = frozenset({".csv", ".json", ".jsonl", ".log", ".txt", ".md",
                         ".html", ".npy", ".npz", ".png"})


def _tree_entry(rel: str, kind: str, mode: str | None = None) -> dict:
    target = paths.resolve(rel)
    info = _file_info(target) if target else None
    return {"path": rel, "kind": kind,  # expected | optional | unexpected
            "label": rel.rsplit("/", 1)[-1],
            "preview": bool(info) and Path(rel).suffix.lower() in PREVIEW_EXT,
            "mode": mode,  # the UI prefixes the label with this when set
            "status": "present" if info else "missing",
            "size": info["size"] if info else None,
            "mtime": info["mtime"] if info else None}


def _listdir_extra(dirpath: Path, base: str, known: set[str], kind: str,
                   mode: str | None = None) -> list[dict]:
    if not dirpath.is_dir():
        return []
    return [_tree_entry(f"{base}/{p.name}", kind, mode)
            for p in sorted(dirpath.iterdir()) if p.is_file() and p.name not in known]


def _node(name: str, entries: list[dict], children: list[dict] | None = None) -> dict:
    """A tree node with counts rolled up: n_expected/n_present for the schema, n_files for what is on disk."""
    children = children or []
    exp = [e for e in entries if e["kind"] == "expected"]
    return {
        "name": name,
        "entries": entries,
        "children": children,
        "n_expected": len(exp) + sum(c["n_expected"] for c in children),
        "n_present": sum(e["status"] == "present" for e in exp) + sum(c["n_present"] for c in children),
        "n_files": sum(e["status"] == "present" for e in entries) + sum(c["n_files"] for c in children),
    }


def file_tree() -> dict:
    """Artifacts grouped by kind, then by pipeline; each pipeline's table holds all three modes."""
    ds = [_tree_entry(f"{paths.DATA_PREFIX}/{cfg['data']}", "expected") for cfg in PIPELINES.values()]
    ds += _listdir_extra(paths.DATA_DIR, paths.DATA_PREFIX,
                         {cfg["data"] for cfg in PIPELINES.values()}, "unexpected")

    emb_known = {f"{m}_prompt.npy" for m in EMB_MODELS}
    kinds: dict[str, list[dict]] = {k: [] for k in ("Embeddings", "Results", "Figures", "Models")}
    for pipeline, cfg in PIPELINES.items():
        bucket: dict[str, list[dict]] = {k: [] for k in kinds}
        for mode in MODES:
            base = f"{paths.RUNS_PREFIX}/{pipeline}/{mode}"
            d = run_dir(pipeline, mode)
            bucket["Embeddings"] += [_tree_entry(f"{base}/embeddings/{m}_prompt.npy", "expected", mode)
                                     for m in EMB_MODELS]
            bucket["Embeddings"] += _listdir_extra(d / "embeddings", f"{base}/embeddings",
                                                   emb_known, "unexpected", mode)
            bucket["Results"] += [_tree_entry(f"{base}/results/{n}", "expected", mode)
                                  for n in EXPECTED_RESULTS]
            bucket["Results"] += _listdir_extra(d / "results", f"{base}/results",
                                                set(EXPECTED_RESULTS), "unexpected", mode)
            bucket["Figures"] += _listdir_extra(d / "figures", f"{base}/figures", set(), "optional", mode)
            bucket["Models"] += _listdir_extra(d / "models", f"{base}/models", set(), "optional", mode)
        short = cfg["label"].split(" (")[0]  # drop the parenthetical; the tree is narrow
        for kind, entries in bucket.items():
            kinds[kind].append(_node(short, entries))

    groups = [_node("Input datasets", ds)]
    groups += [_node(kind, [], children) for kind, children in kinds.items()]
    groups.append(_node("Other", [], _smoke_children()))
    return {"groups": groups}


def _smoke_children() -> list[dict]:
    """Smoke (row-limited) runs land under runs/_smoke so they cannot overwrite a real run; list them here."""
    smoke_root = paths.RUNS_DIR / paths.SMOKE_SUBDIR
    smoke = []
    if smoke_root.is_dir():
        for sub_dir in sorted(d for d in smoke_root.rglob("*") if d.is_dir()):
            rel = sub_dir.relative_to(smoke_root).as_posix()
            for e in _listdir_extra(sub_dir, f"{paths.RUNS_PREFIX}/{paths.SMOKE_SUBDIR}/{rel}",
                                    set(), "optional"):
                e["label"] = f"{rel}/{e['label']}"
                smoke.append(e)
    return [_node("Smoke runs", smoke)] if smoke else []
