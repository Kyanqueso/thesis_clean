import argparse
import warnings
import faulthandler
import json
import platform
import sys
import time
from pathlib import Path

try:                      # Unix only; gives the kernel's peak-RSS high-water mark
    import resource
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False

# on a native crash (SIGSEGV etc.), dump every thread's stack so the failing line is visible
faulthandler.enable(all_threads=True)
import progress  # src/progress.py: the marker lines the dashboard's bars read
import numpy as np
import pandas as pd
import matplotlib
# file-only backend: Windows' default TkAgg aborts when joblib threads collect its objects
matplotlib.use("Agg")
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

warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42   # every random choice below (split, classifiers, projections, latency rows) uses it

TEST_SIZE = 0.2            # used when the dataset has no build-time split column
ALPHA = 0.05               # family-wise significance level for the Holm-corrected McNemar tests
MODES = ["normal", "user_intent_only", "context_only"]   # ablation modes compared across
# single-row predictions timed per model for the latency distribution
LATENCY_REPEATS = 200
LATENCY_WARMUP = 20
# b + c at or below this uses the exact binomial McNemar test instead of chi-square
MCNEMAR_EXACT_THRESHOLD = 25
DEGRADATION_METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC"]

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def _peak_rss_bytes() -> float:
    """Process peak RSS as the OS tracks it (getrusage on Unix, psutil peak_wset on Windows)."""
    if _HAS_RESOURCE:
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(ru if sys.platform == "darwin" else ru * 1024)
    if HAS_PSUTIL:
        info = psutil.Process().memory_info()
        return float(getattr(info, "peak_wset", info.rss))
    return float("nan")


class _ResourceMeter:
    """Wall time, CPU-seconds and peak-RSS growth over a block; no sampler thread (it caused SIGSEGV)."""

    def __init__(self):
        self._proc = psutil.Process() if HAS_PSUTIL else None
        self.wall_seconds = 0.0
        self.cpu_seconds = float("nan")
        self.peak_rss_mb = float("nan")

    def __enter__(self):
        self._base_peak_rss = _peak_rss_bytes()
        if self._proc:
            ct = self._proc.cpu_times()
            self._cpu0 = ct.user + ct.system
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall_seconds = time.perf_counter() - self._t0
        if self._proc:
            ct = self._proc.cpu_times()
            self.cpu_seconds = (ct.user + ct.system) - self._cpu0
        peak = _peak_rss_bytes()
        if np.isfinite(peak) and np.isfinite(self._base_peak_rss):
            self.peak_rss_mb = max(0.0, (peak - self._base_peak_rss) / 1e6)
        return False


def measure_latency(model, X_test: np.ndarray, repeats: int = LATENCY_REPEATS,
                    seed: int = SEED) -> dict:
    """Single-row inference latency in ms, one request at a time (not batch throughput)."""
    if repeats <= 0 or len(X_test) == 0:
        return {}          # LATENCY_REPEATS = 0 opts out; the columns are simply absent
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X_test), size=repeats + LATENCY_WARMUP)
    # one contiguous (1, d) row per call, so slicing cost is not timed
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


def environment_report() -> dict:
    """Machine facts the cost columns only mean something against."""
    env = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python': platform.python_version(),
        'psutil_available': HAS_PSUTIL,
    }
    if HAS_PSUTIL:
        env['cpu_count_logical'] = psutil.cpu_count(logical=True)
        env['cpu_count_physical'] = psutil.cpu_count(logical=False)
        env['total_ram_gb'] = round(psutil.virtual_memory().total / 1e9, 2)
    return env


def get_classifiers() -> dict:
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, max_depth=10),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)),
        "SVM": make_pipeline(StandardScaler(), CalibratedClassifierCV(LinearSVC(dual=False, max_iter=1000, tol=1e-3, random_state=SEED), cv=5, n_jobs=-1))
    }
    if HAS_XGB:
        classifiers["XGBoost"] = XGBClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, eval_metric="logloss", verbosity=0)
    if HAS_LGBM:
        # deterministic + row-wise: the same trees whatever the machine's thread count
        classifiers["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, verbosity=-1,
                                                 deterministic=True, force_row_wise=True)
    return classifiers

def load_data(data_path: Path) -> pd.DataFrame | None:
    if not data_path.is_file():
        print(f"Error: Data file not found at: {data_path}")
        return None
    print(f"Loading data from: {data_path.name}...")
    df = pd.read_json(data_path, lines=True)
    labels = df["label"].values
    print(f"   - Total samples: {len(labels):,}")
    print(f"   - Malicious: {labels.sum():,} ({labels.mean():.1%})")
    print(f"   - Benign: {(~labels.astype(bool)).sum():,} ({1-labels.mean():.1%})")
    return df

def split_by_doc(df: pd.DataFrame, indices: np.ndarray, test_size: float,
                 seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Train/test split: build-time `split` column, else grouped by document text, else stratified rows."""
    # honour the build-time split; re-splitting would hand test attack types back to training
    if "split" in df.columns:
        in_train = (df["split"] == "train").to_numpy()
        print(f"\nBuild-time split: {df.loc[in_train, 'doc_id'].nunique():,} train / "
              f"{df.loc[~in_train, 'doc_id'].nunique():,} test documents"
              f" — test attack types were held out when the dataset was built.")
        return indices[in_train], indices[~in_train]

    if "doc_id" not in df.columns:
        return train_test_split(indices, test_size=test_size,
                                stratify=df["label"].values, random_state=seed)

    # group by clean text, not doc_id, so twins and repeated documents stay on one side
    key = df["original_context"] if "original_context" in df.columns else df["doc_id"]
    groups = pd.factorize(key)[0]
    uniq = np.unique(groups)
    train_groups = train_test_split(uniq, test_size=test_size, random_state=seed)[0]
    in_train = np.isin(groups, train_groups)
    print(f"\nGrouped split: {len(train_groups):,} train / {len(uniq) - len(train_groups):,}"
          f" test distinct documents — no document text on both sides.")
    return indices[in_train], indices[~in_train]

DECISION_THRESHOLD = 0.5


def quality_metrics(y_true, y_prob, threshold: float = DECISION_THRESHOLD) -> dict:
    """Every detection-quality metric; shared by the run table and the ablation tables."""
    y_pred = (y_prob >= threshold).astype(int)

    # labels=[0,1] keeps a 2x2 matrix even when a model predicts only one class
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'PR-AUC': average_precision_score(y_true, y_prob),
        'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        'Specificity': float(tn / (tn + fp)) if (tn + fp) else float('nan'),
        'FPR': float(fp / (fp + tn)) if (fp + tn) else float('nan'),
        'FNR': float(fn / (fn + tp)) if (fn + tp) else float('nan'),
    }


def train_evaluate(X_train, y_train, X_test, y_test, classifier, name):
    """Train one classifier and measure quality and cost; returns (metrics, model, y_prob, y_pred)."""
    print(f"    - Training and evaluating: {name}")

    with _ResourceMeter() as fit_res:
        classifier.fit(X_train, y_train)

    # batch inference over the whole test set measures throughput, not latency
    with _ResourceMeter() as infer_res:
        y_prob = classifier.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= DECISION_THRESHOLD).astype(int)

    metrics = {'Classifier': name}
    metrics.update(quality_metrics(y_test, y_prob))
    metrics.update({
        'Train-Time(s)': fit_res.wall_seconds,
        'Train-CPU(s)': fit_res.cpu_seconds,
        'Train-PeakRSS(MB)': fit_res.peak_rss_mb,
        'Inference-Time-Total(s)': infer_res.wall_seconds,
        'Inference-CPU(s)': infer_res.cpu_seconds,
        'Inference-PeakRSS(MB)': infer_res.peak_rss_mb,
        'Test-Samples': len(X_test),
    })
    metrics.update(measure_latency(classifier, X_test))
    return metrics, classifier, y_prob, y_pred


# McNemar needs models scored on the same test rows: within a run, or across one pipeline's modes

def mcnemar_test(correct_a: np.ndarray, correct_b: np.ndarray) -> dict:
    """Paired McNemar test: exact binomial for few discordant pairs, else chi-square with Yates."""
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
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
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


# cross-ablation reads <modes_root>/<mode>/results/predictions.npz; each finished mode rewrites it
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
    """Keep the modes whose test rows match the reference mode's (the first of mode_order on disk)."""
    if not loaded:
        return {}, []
    ref = loaded[next((m for m in (mode_order or []) if m in loaded), sorted(loaded)[0])]

    def same_split(a: dict) -> bool:
        return (np.array_equal(a['test_idx'].astype(np.int64), ref['test_idx'].astype(np.int64))
                and np.array_equal(a['y_true'].astype(np.int8), ref['y_true'].astype(np.int8)))

    kept = {m: d for m, d in loaded.items() if same_split(d)}
    skipped = [m for m in loaded if m not in kept]
    return kept, skipped


def compare_across_modes(loaded: dict, alpha: float = 0.05
                         ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """McNemar and degradation (reference - variant) across modes, matched on embedding x classifier."""
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


def plot_dimensionality_reduction(X, y, emb_name, out_dir):
    """PCA, t-SNE and UMAP plots on up to 5,000 sampled rows."""
    print(f"  Creating dimensionality reduction plots for {emb_name}...")

    with plt.style.context('default'):
        sample_size = min(len(X), 5000)
        # own generator: the same rows for every embedding, whatever ran before
        indices = np.random.default_rng(SEED).choice(len(X), sample_size, replace=False)
        X_sample, y_sample = X[indices], y[indices]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout="constrained")

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

        print("    - Computing t-SNE...")
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=min(30, sample_size - 1))
        X_tsne = tsne.fit_transform(X_sample)
        axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap="RdYlBu", alpha=0.6, s=20)
        axes[1].set_title(f't-SNE - {emb_name.upper()}')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].grid(True, alpha=0.3)

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

        cbar = fig.colorbar(
            scatter_plot, ax=axes, orientation="horizontal", location="bottom",
            fraction=0.06, pad=0.1, aspect=40
        )
        cbar.set_label('Label (0: Benign, 1: Malicious)')

        plt.savefig(out_dir / f'{emb_name}_dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_roc_pr_curves(X_test, y_test, models, emb_name, out_dir):
    print(f"  Creating ROC and PR curves for {emb_name}...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax1.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_prob):.3f})')

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ax2.plot(recall, precision, label=f'{name} (AUC = {auc_score(recall, precision):.3f})')

    ax1.plot([0, 1], [0, 1], 'k--', label='Chance')
    ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate', title=f'ROC Curves - {emb_name.upper()}')
    ax1.legend()
    ax2.set(xlabel='Recall', ylabel='Precision', title=f'Precision-Recall Curves - {emb_name.upper()}')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_dir / f'{emb_name}_roc_pr_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(cms: dict, emb_name: str, out_dir: Path):
    """One confusion matrix per classifier: raw counts, shaded by row-normalised rate."""
    print(f"  Creating confusion matrices for {emb_name}...")
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
    print("  Creating final performance summary plots...")
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

def main():
    parser = argparse.ArgumentParser(
        description="Train, evaluate and visualise classifiers on one run's embeddings.")
    parser.add_argument("--data", type=Path, required=True, help="The run's dataset (.jsonl).")
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="runs/<pipeline>/<mode>: reads embeddings/, writes results/, figures/ and models/.")
    args = parser.parse_args()
    EMBED_DIR = args.run_dir / "embeddings"
    RESULTS_DIR = args.run_dir / "results"
    FIGURES_DIR = args.run_dir / "figures"
    MODELS_DIR = args.run_dir / "models"
    for d in (RESULTS_DIR, FIGURES_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    if df is None:
        return 1
    y = df["label"].values
    indices = np.arange(len(y))

    train_idx, test_idx = split_by_doc(df, indices, TEST_SIZE, SEED)
    print(f"\nSplitting data: {len(train_idx):,} train, {len(test_idx):,} test samples "
          f"({y[test_idx].mean():.1%} malicious in test).")

    y_train, y_test = y[train_idx], y[test_idx]

    available_embeddings = {p.stem.replace('_prompt', ''): p for p in EMBED_DIR.glob('*_prompt.npy')}
    if not available_embeddings:
        print(f"Error: no *_prompt.npy embeddings in {EMBED_DIR}. Run generate_embeddings.py first.")
        return 1

    if not HAS_PSUTIL:
        print("\nWarning: psutil is not installed; the CPU and memory columns will be NaN.")

    all_results = []
    classifiers = get_classifiers()
    prediction_store = {}
    prediction_labels = {}
    emb_dims = {}

    n_fits, fits_done = len(available_embeddings) * len(classifiers), 0
    for emb_name, emb_path in available_embeddings.items():
        print(f"\n{'='*70}\nProcessing Embeddings: {emb_name.upper()}\n{'='*70}")

        embeddings = np.load(emb_path)
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        emb_dims[emb_name] = int(embeddings.shape[1])

        progress.mark("project", emb_name)
        plot_dimensionality_reduction(embeddings, y, emb_name, FIGURES_DIR)

        trained_models = {}
        confusion_matrices = {}
        for clf_name, classifier in classifiers.items():
            progress.mark("train", f"{emb_name} · {clf_name}", fits_done, n_fits)
            fits_done += 1
            result, trained_model, y_prob, y_pred = train_evaluate(
                X_train, y_train, X_test, y_test, classifier, clf_name)
            result['Embeddings'] = emb_name.upper()
            all_results.append(result)
            trained_models[clf_name] = trained_model
            confusion_matrices[clf_name] = np.array([[result['TN'], result['FP']],
                                                     [result['FN'], result['TP']]])
            prediction_store[f"{emb_name}__{clf_name}"] = y_prob.astype(np.float32)
            prediction_labels[f"{emb_name}__{clf_name}"] = y_pred.astype(np.int8)
            joblib.dump(trained_model, MODELS_DIR / f"{emb_name}_{clf_name}.joblib")

        plot_roc_pr_curves(X_test, y_test, trained_models, emb_name, FIGURES_DIR)
        plot_confusion_matrices(confusion_matrices, emb_name, FIGURES_DIR)

    progress.mark("report", "tables, significance tests, figures")

    if not all_results:
        print("\nNo models were trained. Check for issues during the process.")
        return 1

    results_df = pd.DataFrame(all_results).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)

    # batch throughput (not latency), kept under its original name so older runs stay comparable
    results_df['Inference-Time-per-Sample(ms)'] = (results_df['Inference-Time-Total(s)'] / results_df['Test-Samples']) * 1000

    display_cols = ['Embeddings', 'Classifier', 'Accuracy', 'Precision', 'Recall',
                    'F1-Score', 'ROC-AUC', 'PR-AUC', 'Latency-P50(ms)', 'Train-Time(s)']
    print("\nResults (best ROC-AUC first):")
    print(results_df[[c for c in display_cols if c in results_df]].to_string(index=False, float_format="%.4f"))

    print("\nSaving results...")
    results_df.to_csv(RESULTS_DIR / 'full_evaluation_results.csv', index=False)
    print(f"   Full results saved to: {RESULTS_DIR / 'full_evaluation_results.csv'}")

    # per-sample probabilities, read by the dashboard and by the cross-ablation McNemar
    np.savez_compressed(RESULTS_DIR / 'predictions.npz',
                        y_true=y_test.astype(np.int8), test_idx=test_idx, **prediction_store)
    print(f"   Per-sample predictions saved to: {RESULTS_DIR / 'predictions.npz'}")

    within = mcnemar_within_run(prediction_labels, y_test, alpha=ALPHA)
    if not within.empty:
        within.to_csv(RESULTS_DIR / 'mcnemar_within_run.csv', index=False)
        n_sig = int(within['significant'].sum())
        print(f"   McNemar (within run): {len(within)} pairs, {n_sig} significant at "
              f"Holm alpha={ALPHA} → {RESULTS_DIR / 'mcnemar_within_run.csv'}")
        top = within.head(3)
        for _, r in top.iterrows():
            print(f"      · {r['Model-A']} vs {r['Model-B']}: "
                  f"Δacc {r['Acc-Diff(A-B)']:+.4f}, b={r['b']} c={r['c']}, "
                  f"p_holm={r['p_holm']:.3e} ({r['method']})")

    # cross-ablation: the pipeline folder holds every mode's run; each finished mode rewrites the tables
    modes_root = args.run_dir.parent
    loaded = _load_mode_predictions(modes_root, MODES)
    aligned, skipped = _aligned_modes(loaded, MODES)
    for mode in skipped:
        print(f"   {mode}: test rows differ from the reference mode — excluded from the "
              f"paired comparison (rebuild it from the same dataset/seed to include it).")
    if len(aligned) < 2:
        print(f"   Cross-ablation comparison needs 2+ finished modes under {modes_root}; "
              f"found {len(aligned)} ({', '.join(sorted(aligned)) or 'none'}). "
              f"It is written again by each later mode, so the last one leaves the full matrix.")
    else:
        ablation_mcnemar, ablation_degradation = compare_across_modes(aligned, alpha=ALPHA)
        mc_path = modes_root / 'ablation_mcnemar.csv'
        deg_path = modes_root / 'ablation_degradation.csv'
        ablation_mcnemar.to_csv(mc_path, index=False)
        ablation_degradation.to_csv(deg_path, index=False)
        print(f"   Cross-ablation McNemar over {len(aligned)} modes "
              f"({', '.join(sorted(aligned))}): {len(ablation_mcnemar)} paired tests, "
              f"{int(ablation_mcnemar['significant'].sum())} significant → {mc_path}")
        print(f"   Absolute performance degradation → {deg_path}")
        headline = (ablation_degradation
                    .groupby(['Reference-Mode', 'Variant-Mode'])[[f'Deg-{m}' for m in DEGRADATION_METRICS]]
                    .mean())
        print("\nAbsolute performance degradation, mean over matched configurations "
              "(percentage points; positive = variant is worse):")
        print((headline * 100).round(2).to_string())

    run_meta = {
        'data_file': str(args.data),
        'n_samples': int(len(y)),
        'n_malicious': int(y.sum()),
        'n_benign': int(len(y) - y.sum()),
        'n_train': int(len(train_idx)),
        'n_test': int(len(test_idx)),
        'test_size': TEST_SIZE,
        'seed': SEED,
        'embedding_dims': emb_dims,
        'classifiers': list(classifiers.keys()),
        'decision_threshold': DECISION_THRESHOLD,
        'latency_repeats': LATENCY_REPEATS,
        'alpha': ALPHA,
        'environment': environment_report(),
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(RESULTS_DIR / 'run_meta.json', 'w') as f:
        json.dump(run_meta, f, indent=4, default=_json_safe)
    print(f"   Run metadata saved to: {RESULTS_DIR / 'run_meta.json'}")

    plot_performance_summary(results_df, FIGURES_DIR)
    print(f"   Performance plots saved in: {FIGURES_DIR}/")

    print("\nPipeline completed successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
