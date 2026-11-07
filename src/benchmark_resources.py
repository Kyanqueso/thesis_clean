"""Resource benchmark for the dashboard's Resources section: run it on the benchmark PC.

Measures, for the normal-mode runs of each pipeline:
  embedding stage  (local models only)  peak VRAM and GPU utilisation while embedding a
                                        sample of prompts on the GPU
  classifier stage (every embedding)    training time, CPU time, peak RAM, single-prompt
                                        latency and the saved model's size

Every measurement runs in a fresh subprocess, so "peak RAM" is the true peak of a process
that loads the embeddings and trains that one classifier (Windows cannot reset a running
process's peak, so measuring several classifiers in one process would mix them up).

Results merge into <runs>/resource_benchmark.json; re-running one part keeps the rest.

    venv\\Scripts\\python.exe src\\benchmark_resources.py
    venv\\Scripts\\python.exe src\\benchmark_resources.py --pipelines pipeline3_organic_injected --stages classify
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import psutil

SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
sys.path.insert(0, str(ROOT_DIR))
from pipeline_paths import DATA_DIR, DATASETS, RUNS_DIR  # noqa: E402

OUT_FILE = RUNS_DIR / "resource_benchmark.json"
LOCAL_MODELS = ("minilm", "qwen3")      # run on this machine's GPU; openai runs on OpenAI's servers
ALL_MODELS = ("minilm", "openai", "qwen3")
SAMPLE_SEED = 42
GPU_POLL_S = 0.05


def _peak_rss_mb() -> float:
    """This process's lifetime peak RSS (fresh subprocess, so it is the peak of this one job)."""
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (ru if sys.platform == "darwin" else ru * 1024) / 1e6
    except ImportError:
        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 1e6


def machine_report() -> dict:
    out = {"cpu": platform.processor() or platform.machine(),
           "cpu_cores": psutil.cpu_count(logical=False),
           "cpu_threads": psutil.cpu_count(logical=True),
           "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
           "os": platform.platform(terse=True)}
    if sys.platform == "win32":
        try:   # platform.processor() is only a family code on Windows
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            out["cpu"] = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out["gpu"] = p.name
            out["gpu_vram_gb"] = round(p.total_memory / 1e9, 1)
    except Exception:
        pass
    return out


# ---------------- workers (one fresh process each) ----------------

class _GpuSampler(threading.Thread):
    """Polls whole-GPU utilisation via NVML until stopped."""

    def __init__(self):
        super().__init__(daemon=True)
        import pynvml
        pynvml.nvmlInit()
        self._nvml = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(0)
        self._stop = threading.Event()
        self.util = []

    def run(self):
        while not self._stop.is_set():
            self.util.append(self._nvml.nvmlDeviceGetUtilizationRates(self._h).gpu)
            time.sleep(GPU_POLL_S)

    def stop(self):
        self._stop.set()
        self.join()


def worker_embed(model_key: str, texts_file: Path, out_file: Path) -> int:
    import torch
    from sentence_transformers import SentenceTransformer
    import generate_embeddings as ge

    cfg = ge.MODELS[model_key]
    texts = json.loads(texts_file.read_text(encoding="utf-8"))
    kwargs = {"torch_dtype": getattr(torch, cfg["dtype"])} if cfg.get("dtype") else None
    model = SentenceTransformer(cfg["model_id"], device="cuda", model_kwargs=kwargs)
    if cfg.get("max_seq_length"):
        model.max_seq_length = cfg["max_seq_length"]
    bs = cfg["batch_size"]

    model.encode(texts[:bs], batch_size=bs)            # warm-up: CUDA kernels, allocator
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    try:
        sampler = _GpuSampler()
    except Exception as e:                              # no NVML: VRAM still works
        print(f"    GPU utilisation not measured ({type(e).__name__}: {e}). pip install nvidia-ml-py")
        sampler = None
    if sampler:
        sampler.start()
    model.encode(texts, batch_size=bs)
    torch.cuda.synchronize()
    if sampler:
        sampler.stop()

    util = sampler.util if sampler else []
    out = {"device": torch.cuda.get_device_name(0),
           "n_prompts": len(texts), "batch_size": bs,
           "vram_peak_mb": torch.cuda.max_memory_reserved() / 1e6,
           "vram_alloc_peak_mb": torch.cuda.max_memory_allocated() / 1e6,
           "gpu_util_mean": float(np.mean(util)) if util else None,
           "gpu_util_max": float(np.max(util)) if util else None,
           "peak_ram_mb": _peak_rss_mb()}
    out_file.write_text(json.dumps(out), encoding="utf-8")
    return 0


def worker_classify(emb_file: Path, split_file: Path, clf_name: str, out_file: Path) -> int:
    import joblib
    import train_evaluate_visualize as tev

    z = np.load(split_file)
    fit_idx, test_idx, y = z["fit_idx"], z["test_idx"], z["y"]
    emb = np.load(emb_file)
    X_fit, X_test = emb[fit_idx], emb[test_idx]
    del emb

    clf = tev.get_classifiers()[clf_name]
    if clf_name == "SVM":   # the same solver choice as the training run
        dual = len({row.tobytes() for row in X_fit}) < X_fit.shape[1]
        clf.set_params(calibratedclassifiercv__estimator__dual=dual)

    proc = psutil.Process()
    ram_before_fit = proc.memory_info().rss / 1e6
    ct0 = proc.cpu_times()
    t0 = time.perf_counter()
    clf.fit(X_fit, y[fit_idx])
    fit_s = time.perf_counter() - t0
    ct1 = proc.cpu_times()
    peak = _peak_rss_mb()

    lat = tev.measure_latency(clf, X_test)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "model.joblib"
        joblib.dump(clf, p)
        model_mb = p.stat().st_size / 1e6

    out = {"fit_s": fit_s,
           "cpu_s": (ct1.user + ct1.system) - (ct0.user + ct0.system),
           "peak_ram_mb": peak,
           "ram_before_fit_mb": ram_before_fit,
           "p50_ms": lat.get("Latency-P50(ms)"),
           "p99_ms": lat.get("Latency-P99(ms)"),
           "model_mb": model_mb,
           "n_fit": int(len(fit_idx))}
    out_file.write_text(json.dumps(out), encoding="utf-8")
    return 0


# ---------------- orchestrator ----------------

def _run_worker(args: list[str]) -> dict | None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "out.json"
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), *args, "--_out", str(out)])
        if r.returncode != 0 or not out.is_file():
            print(f"    failed (exit code {r.returncode})")
            return None
        return json.loads(out.read_text(encoding="utf-8"))


def _load(path: Path) -> dict:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        d = {}
    d.setdefault("embedding", {})
    d.setdefault("classifier", {})
    return d


def _save(data: dict):
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pipelines", help=f"Comma-separated subset of {', '.join(DATASETS)}. Default: all.")
    ap.add_argument("--models", help=f"Comma-separated subset of {', '.join(ALL_MODELS)}. Default: all.")
    ap.add_argument("--stages", default="embed,classify", help="embed, classify, or both (default).")
    ap.add_argument("--classifiers", help="Comma-separated classifier names. Default: all.")
    ap.add_argument("--n", type=int, default=1000, help="Prompts embedded per pipeline for the GPU numbers.")
    ap.add_argument("--_worker", help=argparse.SUPPRESS)
    ap.add_argument("--_a", help=argparse.SUPPRESS)
    ap.add_argument("--_b", help=argparse.SUPPRESS)
    ap.add_argument("--_c", help=argparse.SUPPRESS)
    ap.add_argument("--_out", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._worker == "embed":
        return worker_embed(args._a, Path(args._b), Path(args._out))
    if args._worker == "classify":
        return worker_classify(Path(args._a), Path(args._b), args._c, Path(args._out))

    pipes = [p.strip() for p in args.pipelines.split(",")] if args.pipelines else list(DATASETS)
    models = [m.strip() for m in args.models.split(",")] if args.models else list(ALL_MODELS)
    stages = {s.strip() for s in args.stages.split(",")}
    bad = [p for p in pipes if p not in DATASETS] + [m for m in models if m not in ALL_MODELS]
    if bad or not stages <= {"embed", "classify"}:
        print(f"Unknown: {', '.join(bad) or args.stages}")
        return 1

    import pandas as pd
    import train_evaluate_visualize as tev
    import generate_embeddings as ge
    classifiers = ([c.strip() for c in args.classifiers.split(",")] if args.classifiers
                   else list(tev.get_classifiers()))

    machine = machine_report()
    print(f"Machine: {machine.get('cpu')}, {machine.get('ram_gb')} GB RAM, "
          f"{machine.get('gpu', 'no CUDA GPU')}")
    data = _load(OUT_FILE)
    data["machine"] = machine

    cuda = "gpu" in machine
    if "embed" in stages and any(m in LOCAL_MODELS for m in models) and not cuda:
        print("\nNo CUDA GPU visible to PyTorch, so the GPU numbers are skipped.\n"
              "Install the CUDA build of torch (README, step 2, 'NVIDIA GPUs only') and run again.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for pipe in pipes:
            data_file = DATA_DIR / DATASETS[pipe]
            emb_dir = RUNS_DIR / pipe / "normal" / "embeddings"
            if not data_file.is_file():
                print(f"\n{pipe}: dataset {data_file} not found - skipped")
                continue
            print(f"\n=== {pipe} ===")
            df = pd.read_json(data_file, lines=True)
            stamp = time.strftime("%Y-%m-%dT%H:%M:%S")

            if "embed" in stages and cuda:
                rng = np.random.default_rng(SAMPLE_SEED)
                pick = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
                texts = [f"{i}{ge.SEPARATOR}{c}" for i, c in
                         zip(df["user_intent"].astype(str).iloc[pick], df["context"].astype(str).iloc[pick])]
                tf = tmp / f"{pipe}_texts.json"
                tf.write_text(json.dumps(texts), encoding="utf-8")
                for m in models:
                    if m not in LOCAL_MODELS:
                        continue
                    print(f"  embed, {m} ({len(texts)} prompts)")
                    res = _run_worker(["--_worker", "embed", "--_a", m, "--_b", str(tf)])
                    if res:
                        data["embedding"][f"{pipe}|{m}"] = {**res, "measured_at": stamp, "machine": machine}
                        _save(data)
                        print(f"    VRAM {res['vram_peak_mb']:.0f} MB, GPU "
                              f"{res['gpu_util_mean'] if res['gpu_util_mean'] is not None else '?'}% avg")

            if "classify" in stages:
                y = df["label"].values
                train_idx, test_idx = tev.split_by_doc(df, np.arange(len(y)), tev.TEST_SIZE, tev.SEED)
                fit_idx, _ = tev.split_rows(df, train_idx, tev.CAL_SIZE, tev.SEED)
                sf = tmp / f"{pipe}_split.npz"
                np.savez(sf, fit_idx=fit_idx, test_idx=test_idx, y=y)
                for m in models:
                    ef = emb_dir / f"{m}_prompt.npy"
                    if not ef.is_file():
                        print(f"  classify, {m}: no {ef.name} in {emb_dir} - skipped")
                        continue
                    if np.load(ef, mmap_mode="r").shape[0] != len(y):
                        print(f"  classify, {m}: {ef.name} has a different row count than the dataset - skipped")
                        continue
                    for clf in classifiers:
                        print(f"  classify, {m}, {clf}")
                        res = _run_worker(["--_worker", "classify", "--_a", str(ef), "--_b", str(sf), "--_c", clf])
                        if res:
                            data["classifier"][f"{pipe}|{m}|{clf}"] = {**res, "measured_at": stamp,
                                                                       "machine": machine}
                            _save(data)
                            print(f"    fit {res['fit_s']:.1f}s, peak RAM {res['peak_ram_mb']:.0f} MB, "
                                  f"p50 {res['p50_ms']:.2f} ms")

    print(f"\nSaved to {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
