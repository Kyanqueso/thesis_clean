#!/usr/bin/env python
"""
Model Training, Evaluation, and Visualization Script
=====================================================
This script trains machine learning classifiers on pre-generated text embeddings.
It evaluates their performance, generates comprehensive visualizations (including
dimensionality reduction plots and performance curves), and saves all results.

Metrics reported per (embedding x classifier) configuration:

  Detection quality  accuracy, precision, recall, specificity, F1, ROC-AUC,
                     PR-AUC (average precision), and the raw confusion matrix
                     (TN/FP/FN/TP) at the 0.5 decision threshold.
  Degradation        absolute performance degradation ACROSS THE ABLATION RUNS
                     (normal / user_intent_only / context_only), matched on
                     (embedding, classifier): how much accuracy is lost when the
                     model sees only the context, or only the intent?
  Significance       McNemar tests, Holm-Bonferroni corrected, in two places —
                     between every configuration inside this run, and across the
                     ablation modes for each matched configuration. Both are
                     legitimate because the compared models score the same test
                     rows; the cross-mode one asserts that rather than assuming.
  Cost               wall time (time.perf_counter), CPU-seconds and peak RSS
                     (psutil), GPU VRAM attributable to this process (pynvml),
                     and a single-sample inference latency distribution.
"""

import argparse
import warnings
import json
import os
import platform
import threading
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve, auc as auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# --- Optional Imports ---
# Attempt to import optional libraries and set flags indicating their availability.
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# `nvidia-ml-py` and the legacy `pynvml` package both expose the module as
# `pynvml`, so one import covers either install.
try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42
np.random.seed(SEED)

# Single-sample latency probe: how many one-row predictions to time per model.
LATENCY_REPEATS = 200
LATENCY_WARMUP = 20
# b + c at or below this uses the exact binomial McNemar test rather than the
# chi-square approximation (the statsmodels convention; matches thesis4/src/stats.py).
MCNEMAR_EXACT_THRESHOLD = 25
# Metrics the degradation tables are computed over.
DEGRADATION_METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]

# --- Project Structure Paths ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_EMBED_DIR = ROOT_DIR / "embeddings"
DEFAULT_RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_FIGURES_DIR = ROOT_DIR / "figures"

# --- Plotting Style ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# --- Resource Measurement -------------------------------------------------
# Three separate costs, measured separately because they answer different
# questions: wall time (how long did the user wait), CPU-seconds (how much
# compute was actually burned — an n_jobs=-1 forest burns far more than its
# wall-clock suggests), and peak memory (what does the box need to have).

class _GpuProbe:
    """VRAM attributable to THIS process, via NVML.

    `nvmlDeviceGetMemoryInfo` reports the whole card, which on a shared or
    desktop GPU includes everybody else's allocations — useless as a footprint.
    `nvmlDeviceGetComputeRunningProcesses` breaks it down per PID, which is the
    number we actually want. Two ways that degrades:

      - the driver may refuse to attribute memory per process (Windows in WDDM
        mode returns usedGpuMemory=None); we then fall back to device-total and
        say so in `mode`, because a contaminated number labelled as such beats a
        silent zero.
      - a process with no CUDA context is simply absent from the list, which
        correctly reads as 0 MB. That is the expected result here: every
        classifier in this script is CPU-only, so the pipeline's real VRAM cost
        lives in generate_embeddings.py (Qwen3-4B), not in this stage.
    """

    def __init__(self):
        self.ok = False
        self.mode = "unavailable"
        self.reason = ""
        self.name = None
        self.total_mb = 0.0
        self.attribution_failures = 0
        self._last_good = 0
        self._handles = []
        self._pid = os.getpid()

        if not HAS_PYNVML:
            self.reason = "pynvml is not installed (pip install nvidia-ml-py)"
            return
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            if not self._handles:
                self.reason = "NVML initialised but reports no devices"
                return
            raw_name = pynvml.nvmlDeviceGetName(self._handles[0])
            self.name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            self.total_mb = sum(pynvml.nvmlDeviceGetMemoryInfo(h).total
                                for h in self._handles) / 1e6
        except Exception as e:  # noqa: BLE001 - no driver, no card, permissions
            self.reason = f"{type(e).__name__}: {e}"
            return

        self.mode = "device_total" if self._process_bytes() is None else "per_process"
        self.ok = True

    def _process_bytes(self) -> int | None:
        """Bytes this PID holds across all devices, or None if unattributable."""
        total = 0
        for handle in self._handles:
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            except Exception:  # noqa: BLE001
                return None
            for p in procs:
                if p.pid != self._pid:
                    continue
                if getattr(p, "usedGpuMemory", None) is None:
                    return None      # WDDM: driver will not attribute per-process
                total += int(p.usedGpuMemory)
        return total

    def _device_bytes(self) -> int:
        try:
            return sum(pynvml.nvmlDeviceGetMemoryInfo(h).used for h in self._handles)
        except Exception:  # noqa: BLE001
            return 0

    def used_bytes(self) -> int:
        """Current VRAM reading, always on this probe's fixed accounting basis.

        The mode is decided once, at init, and never changes. Switching basis
        mid-measurement is what makes a number meaningless: a baseline taken
        per-process against a peak taken device-total reports whatever ELSE the
        machine allocated in between — a CPU-only fit "using" 200 MB of VRAM.
        So a transient attribution failure reuses the last good reading (the
        delta then contributes nothing) and is counted instead.
        """
        if not self.ok:
            return 0
        if self.mode == "per_process":
            b = self._process_bytes()
            if b is None:
                self.attribution_failures += 1
                return self._last_good
        else:
            b = self._device_bytes()
        self._last_good = b
        return b

    def describe(self) -> dict:
        return {"available": self.ok, "mode": self.mode, "device": self.name,
                "total_vram_mb": round(self.total_mb, 1),
                "attribution_failures": self.attribution_failures,
                "reason": self.reason or None}


GPU = _GpuProbe()


class _ResourceMeter:
    """Wall time, CPU-seconds, peak RSS growth and peak VRAM growth over a block.

    Peaks need sampling rather than a before/after read: a fit can allocate and
    free inside the block, and only the high-water mark tells you what the
    machine had to supply. A 20 ms daemon thread costs nothing next to a model
    fit and catches anything that lives longer than a couple of frames.

    Every field is a DELTA against block entry, so a value is "what this fit
    cost", not "what the interpreter happened to be holding".
    """

    def __init__(self, interval: float = 0.02):
        self._interval = interval
        self._proc = psutil.Process() if HAS_PSUTIL else None
        self.wall_seconds = 0.0
        self.cpu_seconds = float("nan")
        self.peak_rss_mb = float("nan")
        self.peak_vram_mb = float("nan")

    def __enter__(self):
        self._base_rss = self._peak_rss = self._proc.memory_info().rss if self._proc else 0
        self._base_vram = self._peak_vram = GPU.used_bytes()
        if self._proc:
            ct = self._proc.cpu_times()
            self._cpu0 = ct.user + ct.system
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        self._t0 = time.perf_counter()
        return self

    def _sample(self):
        while not self._stop.wait(self._interval):
            try:
                if self._proc:
                    self._peak_rss = max(self._peak_rss, self._proc.memory_info().rss)
                if GPU.ok:
                    self._peak_vram = max(self._peak_vram, GPU.used_bytes())
            except Exception:  # noqa: BLE001 - process teardown races
                return

    def __exit__(self, *exc):
        self.wall_seconds = time.perf_counter() - self._t0
        self._stop.set()
        self._thread.join(timeout=1.0)
        if self._proc:
            ct = self._proc.cpu_times()
            self.cpu_seconds = (ct.user + ct.system) - self._cpu0
            self.peak_rss_mb = max(0.0, (self._peak_rss - self._base_rss) / 1e6)
        if GPU.ok:
            self.peak_vram_mb = max(0.0, (self._peak_vram - self._base_vram) / 1e6)
        return False


def measure_latency(model, X_test: np.ndarray, repeats: int = LATENCY_REPEATS,
                    seed: int = SEED) -> dict:
    """Single-sample inference latency, in milliseconds.

    Distinct from the batch throughput number: dividing a 14,000-row
    predict_proba by 14,000 measures how well the model VECTORISES, and is
    typically 10-100x optimistic versus answering one request at a time — which
    is what a detector sitting in front of an LLM actually does. Here each
    timed call carries exactly one row.

    time.perf_counter() is the right clock: monotonic, and ~100 ns resolution on
    both Windows and Linux, so a 50 us prediction is still ~500 ticks wide.
    Percentiles are reported because tail latency, not the mean, is what a
    request budget has to absorb.
    """
    if repeats <= 0 or len(X_test) == 0:
        return {}          # --latency-repeats 0 opts out; the columns are simply absent
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X_test), size=repeats + LATENCY_WARMUP)
    # One contiguous (1, d) row per call: reshaping a slice of X_test would time
    # a strided view and add copy cost that belongs to the harness, not the model.
    rows = [np.ascontiguousarray(X_test[i]).reshape(1, -1) for i in idx]

    for row in rows[:LATENCY_WARMUP]:      # JIT/BLAS warmup, first-call caches
        model.predict_proba(row)

    samples = np.empty(repeats, dtype=np.float64)
    for k, row in enumerate(rows[LATENCY_WARMUP:]):
        t0 = time.perf_counter()
        model.predict_proba(row)
        samples[k] = time.perf_counter() - t0

    ms = samples * 1000.0
    return {
        'Latency-Mean(ms)': float(ms.mean()),
        'Latency-P50(ms)': float(np.percentile(ms, 50)),
        'Latency-P95(ms)': float(np.percentile(ms, 95)),
        'Latency-P99(ms)': float(np.percentile(ms, 99)),
        'Latency-Min(ms)': float(ms.min()),
        'Latency-Repeats': int(repeats),
    }


def _json_safe(obj):
    """json.dump default= for the numpy scalars pandas hands back."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj).__name__} is not JSON serialisable")


def _json_records(frame) -> list[dict]:
    """DataFrame/Series -> JSON-safe records.

    NaN has to become null, not the bare `NaN` token Python's json emits by
    default: the dashboard parses these files with JSON.parse, which rejects it.
    Non-finite values are real here — VRAM columns are NaN whenever no GPU was
    found — so this is the normal path, not an edge case.
    """
    if frame is None or len(frame) == 0:
        return []
    records = ([frame.to_dict()] if isinstance(frame, pd.Series)
               else frame.to_dict('records'))
    return [{k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else v)
             for k, v in rec.items()} for rec in records]


def environment_report() -> dict:
    """Machine facts the cost columns only mean something against."""
    env = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python': platform.python_version(),
        'psutil_available': HAS_PSUTIL,
        'pynvml_available': HAS_PYNVML,
        'gpu': GPU.describe(),
    }
    if HAS_PSUTIL:
        env['cpu_count_logical'] = psutil.cpu_count(logical=True)
        env['cpu_count_physical'] = psutil.cpu_count(logical=False)
        env['total_ram_gb'] = round(psutil.virtual_memory().total / 1e9, 2)
    return env


# --- Classifier Definitions ---
def get_classifiers() -> dict:
    """Returns a dictionary of available machine learning classifiers."""
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, max_depth=10),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)),
        "SVM": make_pipeline(StandardScaler(), CalibratedClassifierCV(LinearSVC(dual=False, max_iter=1000, tol=1e-3, random_state=SEED), cv=5, n_jobs=-1))
    }
    if HAS_XGB:
        classifiers["XGBoost"] = XGBClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, eval_metric="logloss", verbosity=0)
    if HAS_LGBM:
        classifiers["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, verbosity=-1)
    return classifiers

# --- Data Handling ---
def load_data(data_path: Path) -> pd.DataFrame | None:
    """Loads the dataset file."""
    if not data_path.is_file():
        print(f"❌ Error: Data file not found at: {data_path}")
        return None
    print(f"🔄 Loading data from: {data_path.name}...")
    df = pd.read_json(data_path, lines=True)
    labels = df["label"].values
    print(f"   - Total samples: {len(labels):,}")
    print(f"   - Malicious: {labels.sum():,} ({labels.mean():.1%})")
    print(f"   - Benign: {(~labels.astype(bool)).sum():,} ({1-labels.mean():.1%})")
    return df

# --- Leakage Guard ---
def split_by_doc(df: pd.DataFrame, indices: np.ndarray, test_size: float,
                 seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick the train/test split, in priority order:

      1. a `split` column stamped at build time (--disjoint-attacks datasets)
      2. grouped on the document TEXT, so every copy of a document stays together
      3. a plain label-stratified row split

    Grouping keeps every row of a document on the same side of the boundary.

    Pipeline 3 uses each document twice — once clean, once injected — so a
    row-level split could put one twin in train and the other in test, and the
    model would be scored on a document it had already memorized. Splitting on
    documents makes that impossible by construction rather than repairing it
    afterwards, and it costs no test rows.

    Datasets without doc_id (pipelines 1 and 2) keep the plain label-stratified
    row split, unchanged. Under the twin design stratification is automatic: each
    document contributes exactly one benign and one malicious row to its side.
    """
    # A dataset built with --disjoint-attacks decided train/test at build time,
    # because each side had to draw from its own BIPIA attack pool. Honor it:
    # re-splitting here would hand test attack types back to the training set.
    if "split" in df.columns:
        in_train = (df["split"] == "train").to_numpy()
        print(f"\n🛡️  Build-time split: {df.loc[in_train, 'doc_id'].nunique():,} train / "
              f"{df.loc[~in_train, 'doc_id'].nunique():,} test documents"
              f" — test attack types were held out when the dataset was built.")
        return indices[in_train], indices[~in_train]

    if "doc_id" not in df.columns:
        return train_test_split(indices, test_size=test_size,
                                stratify=df["label"].values, random_state=seed)

    # Group by the clean TEXT where we have it, not by id. A document's twins
    # share it, and so do repeated documents carrying different doc_ids (CoSQA
    # tops its pool up with repeats). Grouping on doc_id alone would let a
    # repeat sit in train and test at once.
    key = df["original_context"] if "original_context" in df.columns else df["doc_id"]
    groups = pd.factorize(key)[0]
    uniq = np.unique(groups)
    train_groups = train_test_split(uniq, test_size=test_size, random_state=seed)[0]
    in_train = np.isin(groups, train_groups)
    print(f"\n🛡️  Grouped split: {len(train_groups):,} train / {len(uniq) - len(train_groups):,}"
          f" test distinct documents — no document text on both sides.")
    return indices[in_train], indices[~in_train]

# --- Core Logic ---
DECISION_THRESHOLD = 0.5


def quality_metrics(y_true, y_prob, threshold: float = DECISION_THRESHOLD) -> dict:
    """Every detection-quality number, from labels + scores alone.

    Single source of truth: the training loop and the cross-ablation comparison
    both call this, so a run's own table and its degradation table can never
    disagree about what "Accuracy" means.
    """
    y_pred = (y_prob >= threshold).astype(int)

    # labels=[0,1] pins the orientation, so a degenerate model that predicts one
    # class only still returns a 2x2 and unpacks instead of raising.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # PR-AUC two ways. Average precision is the standard estimator and the one
    # to quote. The trapezoidal integral of the PR curve is what this script
    # reported before; it is optimistically biased (linear interpolation between
    # PR points is not achievable), but it is kept so numbers stay comparable
    # against the runs already in runs_from_vastai/.
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'PR-AUC': average_precision_score(y_true, y_prob),
        'PR-AUC-Trapz': auc_score(recall_curve, precision_curve),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        'Specificity': float(tn / (tn + fp)) if (tn + fp) else float('nan'),
        'FPR': float(fp / (fp + tn)) if (fp + tn) else float('nan'),
        'FNR': float(fn / (fn + tp)) if (fn + tp) else float('nan'),
    }


def train_evaluate(X_train, y_train, X_test, y_test, classifier, name,
                   latency_repeats: int = LATENCY_REPEATS):
    """Train one classifier and measure both its detection quality and its cost.

    Returns (metrics, fitted_model, y_prob, y_pred). y_pred is handed back
    because McNemar needs the hard 0/1 decisions, not the scores.
    """
    print(f"    - Training and evaluating: {name}")

    with _ResourceMeter() as fit_res:
        classifier.fit(X_train, y_train)

    # Batch inference over the whole test set -> THROUGHPUT.
    with _ResourceMeter() as infer_res:
        y_prob = classifier.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    metrics = {'Classifier': name}
    metrics.update(quality_metrics(y_test, y_prob))
    metrics.update({
        # --- cost: time, CPU, memory, VRAM ---
        'Train-Time(s)': fit_res.wall_seconds,
        'Train-CPU(s)': fit_res.cpu_seconds,
        'Train-PeakRSS(MB)': fit_res.peak_rss_mb,
        'Train-PeakVRAM(MB)': fit_res.peak_vram_mb,
        'Inference-Time-Total(s)': infer_res.wall_seconds,
        'Inference-CPU(s)': infer_res.cpu_seconds,
        'Inference-PeakRSS(MB)': infer_res.peak_rss_mb,
        'Inference-PeakVRAM(MB)': infer_res.peak_vram_mb,
        'Test-Samples': len(X_test),
    })
    metrics.update(measure_latency(classifier, X_test, repeats=latency_repeats))
    return metrics, classifier, y_prob, y_pred


# --- Statistical Significance (McNemar) -----------------------------------
# Two models are comparable with McNemar only if they were scored on the SAME
# test rows, in the same order. Two places satisfy that here:
#
#   1. within one run, every embedding x classifier combo shares the run's own
#      train/test split;
#   2. across the three ablation modes of one pipeline, because the split is a
#      deterministic function of (dataset, SEED) and the mode only changes which
#      TEXT gets embedded — never which rows are held out.
#
# (2) is asserted, not assumed: _aligned_modes refuses to pair two runs whose
# test_idx or y_true differ. Different PIPELINES are never comparable this way —
# different datasets, different test sets, no pairing.

def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired McNemar test on two correctness vectors.

    Only the discordant pairs carry information: b = A right / B wrong,
    c = A wrong / B right. Rows both models get right (or both wrong) say
    nothing about which is better, which is exactly why the paired test is
    far more sensitive than comparing two independent accuracies.

    Exact binomial when b + c is small (the chi-square approximation is unsafe
    there), chi-square with Yates' continuity correction otherwise.
    """
    from scipy.stats import binom, chi2

    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        return {'b': 0, 'c': 0, 'n_discordant': 0, 'statistic': 0.0,
                'p_value': 1.0, 'method': 'exact'}
    if n <= MCNEMAR_EXACT_THRESHOLD:
        p = float(min(1.0, 2.0 * binom.cdf(min(b, c), n, 0.5)))
        return {'b': b, 'c': c, 'n_discordant': n, 'statistic': float(min(b, c)),
                'p_value': p, 'method': 'exact'}
    stat = (abs(b - c) - 1.0) ** 2 / n
    return {'b': b, 'c': c, 'n_discordant': n, 'statistic': float(stat),
            'p_value': float(chi2.sf(stat, 1)), 'method': 'chi2_continuity'}


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values.

    A run compares 15 configurations pairwise = 105 tests; at alpha=0.05 you
    would expect ~5 spurious "significant" results by chance alone. Holm
    controls the family-wise error rate while being uniformly more powerful
    than plain Bonferroni, so reporting raw p-values here would be a defect.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        # step-down: each adjusted p is monotonically non-decreasing in rank
        running = max(running, (m - rank) * p_values[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def _pairwise_mcnemar(entries: list[tuple[str, np.ndarray]], y_true: np.ndarray,
                      alpha: float = 0.05) -> pd.DataFrame:
    """All-pairs McNemar over (label, y_pred) entries sharing one test set."""
    correct = {name: (pred == y_true) for name, pred in entries}
    rows = []
    names = [n for n, _ in entries]
    for i, a in enumerate(names):
        for b_name in names[i + 1:]:
            res = mcnemar_test(correct[a], correct[b_name])
            acc_a, acc_b = correct[a].mean(), correct[b_name].mean()
            rows.append({'Model-A': a, 'Model-B': b_name,
                         'Acc-A': float(acc_a), 'Acc-B': float(acc_b),
                         'Acc-Diff(A-B)': float(acc_a - acc_b), **res})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['p_holm'] = holm_bonferroni(df['p_value'].tolist())
    df['significant'] = df['p_holm'] < alpha
    return df.sort_values('p_holm').reset_index(drop=True)


def mcnemar_within_run(pred_labels: dict, y_true: np.ndarray,
                       alpha: float = 0.05) -> pd.DataFrame:
    """Every embedding x classifier combo of this run against every other."""
    return _pairwise_mcnemar(sorted(pred_labels.items()), y_true, alpha)


# --- Cross-ablation comparison --------------------------------------------
# Layout assumed (written by run_all_pipelines.py):
#     <modes_root>/<mode>/results/predictions.npz
# Each run rewrites the comparison files with every mode that has finished so
# far, so the last mode to complete leaves the full matrix behind.

def _load_mode_predictions(modes_root: Path, modes: list[str]) -> dict:
    out = {}
    for mode in modes:
        npz_path = modes_root / mode / "results" / "predictions.npz"
        if not npz_path.is_file():
            continue
        with np.load(npz_path) as data:
            out[mode] = {k: data[k] for k in data.files}
    return out


def _aligned_modes(loaded: dict, mode_order: list[str] | None = None
                   ) -> tuple[dict, list[str]]:
    """Keep the largest set of modes that share one test set, exactly.

    The guard that makes pairing legitimate: a mode built from a different
    dataset build, a different seed or a different split would silently produce
    a meaningless McNemar table, so it is dropped with a reason instead.

    Modes are bucketed by their (test_idx, y_true) identity and the biggest
    bucket wins — never "whichever mode sorted first". Anchoring on an arbitrary
    reference lets one stale run evict every good one, which is exactly backwards.
    Ties break toward the bucket holding the earliest mode in `mode_order`, so
    the reference condition (normal) wins a 1-1 split.
    """
    if not loaded:
        return {}, []
    order = mode_order or sorted(loaded)

    def rank(mode: str) -> int:
        return order.index(mode) if mode in order else len(order)

    def same_split(a: dict, b: dict) -> bool:
        return (np.array_equal(a['test_idx'].astype(np.int64), b['test_idx'].astype(np.int64))
                and np.array_equal(a['y_true'].astype(np.int8), b['y_true'].astype(np.int8)))

    buckets: list[list[str]] = []
    for mode in sorted(loaded, key=rank):
        for bucket in buckets:
            if same_split(loaded[mode], loaded[bucket[0]]):
                bucket.append(mode)
                break
        else:
            buckets.append([mode])

    winner = max(buckets, key=lambda b: (len(b), -rank(b[0])))
    kept = {m: loaded[m] for m in winner}
    skipped = [m for m in loaded if m not in kept]
    return kept, skipped


def compare_across_modes(loaded: dict, alpha: float = 0.05
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """McNemar + absolute performance degradation across the ablation modes.

    Both tables are matched on (embedding, classifier): the point of an ablation
    is to hold the model fixed and change only the input, so comparing
    OPENAI+SVM(normal) against MINILM+RF(context_only) would answer nothing.

    Degradation is the plain absolute difference reference - variant, in the
    metric's own units (multiply by 100 for percentage points). Positive means
    the variant is WORSE than the reference. Every ordered pair is emitted, so
    you can read normal->context_only and context_only->normal off the same file.
    """
    if len(loaded) < 2:
        return pd.DataFrame(), pd.DataFrame()

    y_true = loaded[sorted(loaded)[0]]['y_true']
    combos = {m: {k for k in d if k not in ('y_true', 'test_idx')} for m, d in loaded.items()}
    shared = sorted(set.intersection(*combos.values()))

    quality = {m: {c: quality_metrics(y_true, loaded[m][c]) for c in shared}
               for m in loaded}

    mc_rows, deg_rows = [], []
    modes = sorted(loaded)
    for combo in shared:
        emb, clf = combo.split("__", 1)
        for i, mode_a in enumerate(modes):
            for mode_b in modes[i + 1:]:
                pred_a = (loaded[mode_a][combo] >= DECISION_THRESHOLD).astype(int)
                pred_b = (loaded[mode_b][combo] >= DECISION_THRESHOLD).astype(int)
                res = mcnemar_test(pred_a == y_true, pred_b == y_true)
                mc_rows.append({
                    'Embeddings': emb.upper(), 'Classifier': clf,
                    'Mode-A': mode_a, 'Mode-B': mode_b,
                    'Acc-A': quality[mode_a][combo]['Accuracy'],
                    'Acc-B': quality[mode_b][combo]['Accuracy'],
                    'Acc-Diff(A-B)': quality[mode_a][combo]['Accuracy']
                                     - quality[mode_b][combo]['Accuracy'],
                    **res})
        for ref in modes:
            for var in modes:
                if ref == var:
                    continue
                row = {'Embeddings': emb.upper(), 'Classifier': clf,
                       'Reference-Mode': ref, 'Variant-Mode': var}
                for metric in DEGRADATION_METRICS:
                    row[f'Deg-{metric}'] = quality[ref][combo][metric] - quality[var][combo][metric]
                deg_rows.append(row)

    mc_df = pd.DataFrame(mc_rows)
    if not mc_df.empty:
        mc_df['p_holm'] = holm_bonferroni(mc_df['p_value'].tolist())
        mc_df['significant'] = mc_df['p_holm'] < alpha
        mc_df = mc_df.sort_values(['Embeddings', 'Classifier', 'Mode-A', 'Mode-B'])
    deg_df = pd.DataFrame(deg_rows)
    if not deg_df.empty:
        deg_df = deg_df.sort_values(['Reference-Mode', 'Variant-Mode',
                                     'Deg-ROC-AUC'], ascending=[True, True, False])
    return mc_df.reset_index(drop=True), deg_df.reset_index(drop=True)


# --- Visualization Functions ---
def plot_dimensionality_reduction(X, y, emb_name, out_dir):
    """Creates and saves PCA, t-SNE, and UMAP plots with the original visual style."""
    print(f"  🖼️  Creating dimensionality reduction plots for {emb_name}...")
    
    # Use the original plot style for this specific function
    with plt.style.context('default'):
        sample_size = min(len(X), 5000)
        indices = np.random.choice(len(X), sample_size, replace=False)
        X_sample, y_sample = X[indices], y[indices]

        # Use constrained_layout to manage spacing for the colorbar
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")
        
        # --- 1) PCA Plot ---
        print("    - Computing PCA...")
        pca = PCA(n_components=2, random_state=SEED)
        X_pca = pca.fit_transform(X_sample)
        scatter_plot = axes[0].scatter(
            X_pca[:, 0], X_pca[:, 1], c=y_sample, cmap="RdYlBu", alpha=0.6, s=20
        )
        axes[0].set_title(
            f'PCA - {emb_name.upper()}\nExplained Variance: {pca.explained_variance_ratio_.sum():.1%}'
        )
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[0].grid(True, alpha=0.3)

        # --- 2) t-SNE Plot ---
        print("    - Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, sample_size - 1))
        X_tsne = tsne.fit_transform(X_sample)
        axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap="RdYlBu", alpha=0.6, s=20)
        axes[1].set_title(f't-SNE - {emb_name.upper()}')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].grid(True, alpha=0.3)

        # --- 3) UMAP Plot ---
        if HAS_UMAP:
            print("    - Computing UMAP...")
            umap_reducer = umap.UMAP(n_components=2, random_state=SEED, n_neighbors=min(15, sample_size - 1))
            X_umap = umap_reducer.fit_transform(X_sample)
            axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y_sample, cmap="RdYlBu", alpha=0.6, s=20)
            axes[2].set_title(f'UMAP - {emb_name.upper()}')
            axes[2].set_xlabel('UMAP 1')
            axes[2].set_ylabel('UMAP 2')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, 'UMAP not available\npip install umap-learn', 
                         ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('UMAP - Not Available')

        # --- Single Colorbar Below All Subplots ---
        cbar = fig.colorbar(
            scatter_plot, ax=axes, orientation="horizontal", location="bottom",
            fraction=0.06, pad=0.1, aspect=40
        )
        cbar.set_label('Label (0: Benign, 1: Malicious)')
        
        # Save the figure
        plt.savefig(out_dir / f'{emb_name}_dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_roc_pr_curves(X_test, y_test, models, emb_name, out_dir):
    """Creates and saves ROC and Precision-Recall curve plots."""
    print(f"  📈 Creating ROC and PR curves for {emb_name}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.3f})')
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ax2.plot(recall, precision, label=f'{name} (AUC = {auc_score(recall, precision):.3f})')

    # Formatting
    ax1.plot([0, 1], [0, 1], 'k--', label='Chance')
    ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=f'ROC Curves - {emb_name.upper()}')
    ax1.legend()
    ax2.set(xlabel='Recall', ylabel='Precision', title=f'Precision-Recall Curves - {emb_name.upper()}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(out_dir / f'{emb_name}_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(cms: dict, emb_name: str, out_dir: Path):
    """One row of confusion matrices — every classifier for one embedding.

    Annotated with raw counts and shaded by ROW-normalised rate, so the two
    error types stay readable even when the classes are not perfectly balanced:
    the top row is what happens to benign traffic (false positives), the bottom
    row what happens to attacks (missed detections).
    """
    print(f"  🧮 Creating confusion matrices for {emb_name}...")
    n = len(cms)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, (clf_name, cm) in zip(axes, cms.items()):
        rates = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        sns.heatmap(rates, annot=cm, fmt=",d", cmap="Blues", vmin=0, vmax=1,
                    cbar=False, square=True, ax=ax,
                    xticklabels=["Benign", "Malicious"],
                    yticklabels=["Benign", "Malicious"])
        ax.set_title(clf_name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(f"Confusion Matrices — {emb_name.upper()} (threshold {DECISION_THRESHOLD})")
    plt.savefig(out_dir / f'{emb_name}_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_performance_summary(df, out_dir):
    """Creates bar plots summarizing model performance across all embeddings."""
    print("  📊 Creating final performance summary plots...")
    metrics = ['Accuracy', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    for ax, metric in zip(axes.flatten(), metrics):
        sns.barplot(data=df, x='Embeddings', y=metric, hue='Classifier', ax=ax)
        ax.set_title(metric)
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=10)
        ax.legend(title='Classifier')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

# --- Main Execution ---
def main():
    """Main function to orchestrate the training, evaluation, and visualization pipeline."""
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and visualize classifiers on text embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=DATA_DIR / "final_training_dataset.jsonl", help="Path to the input dataset (.jsonl).")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion of the dataset to use for testing.")
    parser.add_argument("--embed-dir", type=Path, default=DEFAULT_EMBED_DIR, help="Directory to read embeddings from.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory to write results to.")
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR, help="Directory to write figures to.")
    parser.add_argument("--save-models", action="store_true", help="Persist each trained classifier to --models-dir as a joblib file.")
    parser.add_argument("--models-dir", type=Path, default=None, help="Directory for saved models (default: <results-dir>/../models).")
    parser.add_argument("--skip-projections", action="store_true", help="Skip the PCA/t-SNE/UMAP plots (they dominate runtime).")
    parser.add_argument("--latency-repeats", type=int, default=LATENCY_REPEATS,
                        help="Single-row predictions to time per model for the latency distribution (0 disables).")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Family-wise significance level for the Holm-corrected McNemar tests.")
    parser.add_argument("--modes-root", type=Path, default=None,
                        help="Directory holding <mode>/results/predictions.npz for the ablation modes "
                             "(default: two levels above --results-dir).")
    parser.add_argument("--modes", type=str, default="normal,user_intent_only,context_only",
                        help="Comma-separated ablation modes to compare across.")
    parser.add_argument("--skip-ablation-compare", action="store_true",
                        help="Skip the cross-ablation McNemar and degradation tables.")
    args = parser.parse_args()
    EMBED_DIR = args.embed_dir
    RESULTS_DIR = args.results_dir
    FIGURES_DIR = args.figures_dir
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR = args.models_dir if args.models_dir else RESULTS_DIR.parent / "models"
    if args.save_models:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(args.data)
    if df is None:
        return
    y = df["label"].values
    indices = np.arange(len(y))

    # Split: by doc_id where present (pipeline 3's clean/injected twins),
    # otherwise the plain label-stratified row split.
    train_idx, test_idx = split_by_doc(df, indices, args.test_size, SEED)
    print(f"\nSplitting data: {len(train_idx):,} train, {len(test_idx):,} test samples "
          f"({y[test_idx].mean():.1%} malicious in test).")

    y_train, y_test = y[train_idx], y[test_idx]

    # Discover available embeddings
    available_embeddings = {p.stem.replace('_prompt', ''): p for p in EMBED_DIR.glob('*_prompt.npy')}
    if not available_embeddings:
        print("❌ Error: No embedding files found in the 'embeddings' directory. Please run the generation script first.")
        return

    print(f"\n🖥️  Environment: {'psutil ok' if HAS_PSUTIL else 'psutil MISSING (CPU/RAM columns will be NaN)'}"
          f" · {('GPU ' + str(GPU.name) + f' [{GPU.mode}]') if GPU.ok else 'GPU unavailable: ' + GPU.reason}")

    all_results = []
    classifiers = get_classifiers()
    prediction_store = {}
    prediction_labels = {}
    emb_dims = {}

    # Process each embedding file
    for emb_name, emb_path in available_embeddings.items():
        print(f"\n{'='*70}\n🚀 Processing Embeddings: {emb_name.upper()}\n{'='*70}")
        
        embeddings = np.load(emb_path)
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        emb_dims[emb_name] = int(embeddings.shape[1])

        if not args.skip_projections:
            plot_dimensionality_reduction(embeddings, y, emb_name, FIGURES_DIR)
        
        trained_models = {}
        confusion_matrices = {}
        for clf_name, classifier in classifiers.items():
            result, trained_model, y_prob, y_pred = train_evaluate(
                X_train, y_train, X_test, y_test, classifier, clf_name,
                latency_repeats=args.latency_repeats)
            result['Embeddings'] = emb_name.upper()
            all_results.append(result)
            trained_models[clf_name] = trained_model
            confusion_matrices[clf_name] = np.array([[result['TN'], result['FP']],
                                                     [result['FN'], result['TP']]])
            prediction_store[f"{emb_name}__{clf_name}"] = y_prob.astype(np.float32)
            prediction_labels[f"{emb_name}__{clf_name}"] = y_pred.astype(np.int8)
            if args.save_models:
                joblib.dump(trained_model, MODELS_DIR / f"{emb_name}_{clf_name}.joblib")

        plot_roc_pr_curves(X_test, y_test, trained_models, emb_name, FIGURES_DIR)
        plot_confusion_matrices(confusion_matrices, emb_name, FIGURES_DIR)

    # --- Final Results Processing ---
    if not all_results:
        print("\n❌ No models were trained. Check for issues during the process.")
        return
        
    results_df = pd.DataFrame(all_results).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
    
    # Batch THROUGHPUT (amortised): whole-test-set time divided by rows. Kept
    # under its original name so existing runs stay comparable — but note it is
    # not latency; Latency-P50(ms) is the one-request-at-a-time number.
    results_df['Inference-Time-per-Sample(ms)'] = (results_df['Inference-Time-Total(s)'] / results_df['Test-Samples']) * 1000

    print("\n\n" + "="*110 + "\n" + " " * 40 + "FINAL RESULTS SUMMARY" + "\n" + "="*110)
    display_cols = ['Embeddings', 'Classifier', 'Accuracy', 'Precision', 'Recall',
                    'F1-Score', 'ROC-AUC', 'PR-AUC', 'Latency-P50(ms)']
    print(results_df[[c for c in display_cols if c in results_df]].to_string(index=False, float_format="%.4f"))

    print("\n" + "-"*110 + "\n" + " " * 42 + "CONFUSION MATRICES\n" + "-"*110)
    cm_cols = ['Embeddings', 'Classifier', 'TN', 'FP', 'FN', 'TP', 'Specificity', 'FPR', 'FNR']
    print(results_df[cm_cols].to_string(index=False, float_format="%.4f"))

    print("\n" + "-"*110 + "\n" + " " * 45 + "RESOURCE COST\n" + "-"*110)
    cost_cols = ['Embeddings', 'Classifier', 'Train-Time(s)', 'Train-CPU(s)',
                 'Train-PeakRSS(MB)', 'Train-PeakVRAM(MB)',
                 'Latency-P50(ms)', 'Latency-P95(ms)', 'Latency-P99(ms)',
                 'Inference-Time-per-Sample(ms)']
    print(results_df[[c for c in cost_cols if c in results_df]].to_string(index=False, float_format="%.4f"))
    print("\n" + "="*110)

    # Inference Time Summary
    avg_inference_ms = results_df['Inference-Time-per-Sample(ms)'].mean()
    print(f"\n⏱️  Inference Time Analysis:")
    print(f"   Batch throughput (amortised): {avg_inference_ms:.4f} ms/sample averaged over all configurations.")
    if 'Latency-P50(ms)' in results_df:
        print(f"   Single-sample latency:        median {results_df['Latency-P50(ms)'].median():.4f} ms, "
              f"p99 {results_df['Latency-P99(ms)'].max():.4f} ms (worst configuration).")
    if HAS_PSUTIL:
        print(f"   Peak fit RSS:                 {results_df['Train-PeakRSS(MB)'].max():.1f} MB (worst configuration).")
    if GPU.ok:
        note = ("expected ~0: these classifiers are CPU-only — the pipeline's VRAM cost is the "
                "embedding model, measured in generate_embeddings.py"
                if GPU.mode == "per_process" else
                "DEVICE-TOTAL basis: this driver will not attribute VRAM per process, so the "
                "figure includes other processes on the card")
        print(f"   Peak fit VRAM:                {results_df['Train-PeakVRAM(MB)'].max():.1f} MB "
              f"[{GPU.mode}] — {note}.")
        if GPU.attribution_failures:
            print(f"   ⚠️ NVML failed to attribute VRAM {GPU.attribution_failures} time(s); "
                  f"those samples reused the previous reading rather than switching basis.")
    else:
        print(f"   GPU VRAM:                     not measured ({GPU.reason}).")

    # Display and save best configuration
    best = results_df.iloc[0]
    print(f"\n🏆 Best Configuration: {best['Embeddings']} + {best['Classifier']}")
    print(f"   - ROC-AUC: {best['ROC-AUC']:.4f}, Accuracy: {best['Accuracy']:.4f}, F1-Score: {best['F1-Score']:.4f}")
    print(f"   - Inference Time: {best['Inference-Time-per-Sample(ms)']:.2f}ms per sample")

    # Save results to files
    print("\n💾 Saving results...")
    results_df.to_csv(RESULTS_DIR / 'full_evaluation_results.csv', index=False)
    print(f"   ✔️ Full results saved to: {RESULTS_DIR / 'full_evaluation_results.csv'}")

    # Per-sample test-set probabilities (keys: "{embedding}__{classifier}") for
    # interactive curves / threshold analysis in the dashboard (app/), and the
    # input the cross-ablation McNemar reads out of the sibling mode directories.
    np.savez_compressed(RESULTS_DIR / 'predictions.npz',
                        y_true=y_test.astype(np.int8), test_idx=test_idx, **prediction_store)
    print(f"   ✔️ Per-sample predictions saved to: {RESULTS_DIR / 'predictions.npz'}")

    # --- McNemar within this run: every configuration against every other ---
    within = mcnemar_within_run(prediction_labels, y_test, alpha=args.alpha)
    if not within.empty:
        within.to_csv(RESULTS_DIR / 'mcnemar_within_run.csv', index=False)
        n_sig = int(within['significant'].sum())
        print(f"   ✔️ McNemar (within run): {len(within)} pairs, {n_sig} significant at "
              f"Holm alpha={args.alpha} → {RESULTS_DIR / 'mcnemar_within_run.csv'}")
        top = within.head(3)
        for _, r in top.iterrows():
            print(f"      · {r['Model-A']} vs {r['Model-B']}: "
                  f"Δacc {r['Acc-Diff(A-B)']:+.4f}, b={r['b']} c={r['c']}, "
                  f"p_holm={r['p_holm']:.3e} ({r['method']})")

    # --- Cross-ablation: McNemar + absolute performance degradation ---
    ablation_mcnemar = ablation_degradation = pd.DataFrame()
    if not args.skip_ablation_compare:
        modes_root = args.modes_root or RESULTS_DIR.parent.parent
        mode_list = [m.strip() for m in args.modes.split(",") if m.strip()]
        loaded = _load_mode_predictions(modes_root, mode_list)
        aligned, skipped = _aligned_modes(loaded, mode_list)
        for mode in skipped:
            print(f"   ⚠️ {mode}: test rows differ from the reference mode — excluded from the "
                  f"paired comparison (rebuild it from the same dataset/seed to include it).")
        if len(aligned) < 2:
            print(f"   ℹ️ Cross-ablation comparison needs 2+ finished modes under {modes_root}; "
                  f"found {len(aligned)} ({', '.join(sorted(aligned)) or 'none'}). "
                  f"It is written again by each later mode, so the last one leaves the full matrix.")
        else:
            ablation_mcnemar, ablation_degradation = compare_across_modes(aligned, alpha=args.alpha)
            mc_path = modes_root / 'ablation_mcnemar.csv'
            deg_path = modes_root / 'ablation_degradation.csv'
            ablation_mcnemar.to_csv(mc_path, index=False)
            ablation_degradation.to_csv(deg_path, index=False)
            print(f"   ✔️ Cross-ablation McNemar over {len(aligned)} modes "
                  f"({', '.join(sorted(aligned))}): {len(ablation_mcnemar)} paired tests, "
                  f"{int(ablation_mcnemar['significant'].sum())} significant → {mc_path}")
            print(f"   ✔️ Absolute performance degradation → {deg_path}")
            headline = (ablation_degradation
                        .groupby(['Reference-Mode', 'Variant-Mode'])[[f'Deg-{m}' for m in DEGRADATION_METRICS]]
                        .mean())
            print("\n📉 Absolute performance degradation, mean over matched configurations "
                  "(percentage points; positive = variant is worse):")
            print((headline * 100).round(2).to_string())

    run_meta = {
        'data_file': str(args.data),
        'n_samples': int(len(y)),
        'n_malicious': int(y.sum()),
        'n_benign': int(len(y) - y.sum()),
        'n_train': int(len(train_idx)),
        'n_test': int(len(test_idx)),
        'test_size': args.test_size,
        'seed': SEED,
        'embedding_dims': emb_dims,
        'classifiers': list(classifiers.keys()),
        'models_saved': bool(args.save_models),
        'decision_threshold': DECISION_THRESHOLD,
        'latency_repeats': args.latency_repeats,
        'alpha': args.alpha,
        'environment': environment_report(),
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(RESULTS_DIR / 'run_meta.json', 'w') as f:
        json.dump(run_meta, f, indent=4, default=_json_safe)
    print(f"   ✔️ Run metadata saved to: {RESULTS_DIR / 'run_meta.json'}")

    # --- HTML Report Export ---
    # Same results table as full_evaluation_results.csv, in HTML form, for visual
    # comparison across runs (see run_all_pipelines.py for the cross-run aggregate).
    html_path = RESULTS_DIR / 'full_evaluation_results.html'
    html_table = results_df.to_html(index=False, float_format=lambda x: f"{x:.4f}")
    html_doc = (
        "<!-- Generated by src/train_evaluate_visualize.py -->\n"
        "<html><head><meta charset=\"utf-8\"><title>Evaluation Results</title>"
        "<style>body{font-family:sans-serif;margin:2rem;}"
        "table{border-collapse:collapse;width:100%;}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;}"
        "th{background:#f2f2f2;text-align:center;}</style></head>"
        f"<body><h2>Evaluation Results</h2>{html_table}</body></html>\n"
    )
    html_path.write_text(html_doc, encoding="utf-8")
    print(f"   ✔️ HTML report saved to: {html_path}")

    summary = {
        'best_configuration': _json_records(best)[0],
        'all_results': _json_records(results_df),
        'mcnemar_within_run': _json_records(within),
        'ablation_mcnemar': _json_records(ablation_mcnemar),
        'ablation_degradation': _json_records(ablation_degradation),
        'info': {
            'test_size': args.test_size,
            'seed': SEED,
            'decision_threshold': DECISION_THRESHOLD,
            'alpha': args.alpha,
            'average_inference_time_ms': avg_inference_ms,
            'environment': environment_report(),
        }
    }
    with open(RESULTS_DIR / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=4, default=_json_safe)
    print(f"   ✔️ Summary saved to: {RESULTS_DIR / 'evaluation_summary.json'}")

    # Create final summary plots
    plot_performance_summary(results_df, FIGURES_DIR)
    print(f"   ✔️ Performance plots saved in: {FIGURES_DIR}/")
    
    print("\n🎉🎉🎉 Pipeline completed successfully! 🎉🎉🎉")

if __name__ == "__main__":
    main()
