#!/usr/bin/env python
"""
Model Training, Evaluation, and Visualization Script
=====================================================
This script trains machine learning classifiers on pre-generated text embeddings.
It evaluates their performance, generates comprehensive visualizations (including
dimensionality reduction plots and performance curves), and saves all results.
"""

import argparse
import warnings
import json
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
    accuracy_score, f1_score, roc_auc_score, roc_curve,
    precision_recall_curve, auc as auc_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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

# --- Configuration ---
warnings.filterwarnings("ignore", category=UserWarning)
SEED = 42
np.random.seed(SEED)

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

# --- Classifier Definitions ---
def get_classifiers() -> dict:
    """Returns a dictionary of available machine learning classifiers."""
    classifiers = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, max_depth=10),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)),
        "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=SEED))
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
def resolve_leakage(doc_ids: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ensures no doc_id appears on both sides of the split by moving the minority-side
    occurrences of any shared doc_id onto the majority side, then re-verifying until clean."""
    print("\n🛡️  Leakage guard: checking for shared doc_id across train/test split...")
    train_idx, test_idx = list(train_idx), list(test_idx)
    rounds, total_moved = 0, 0

    while True:
        train_docs, test_docs = {}, {}
        for i in train_idx:
            train_docs.setdefault(doc_ids[i], []).append(i)
        for i in test_idx:
            test_docs.setdefault(doc_ids[i], []).append(i)

        shared = set(train_docs) & set(test_docs)
        if not shared:
            break

        rounds += 1
        for doc_id in shared:
            t_rows, v_rows = train_docs[doc_id], test_docs[doc_id]
            if len(t_rows) >= len(v_rows):
                for i in v_rows:
                    test_idx.remove(i)
                    train_idx.append(i)
                    total_moved += 1
            else:
                for i in t_rows:
                    train_idx.remove(i)
                    test_idx.append(i)
                    total_moved += 1

    print(f"   - Resolved after {rounds} round(s), moved {total_moved} row(s). 0 shared doc_id remain between train/test.")
    return np.array(train_idx), np.array(test_idx)

# --- Core Logic ---
def train_evaluate(X_train, y_train, X_test, y_test, classifier, name):
    """Trains a single classifier and returns its performance metrics and the trained model."""
    print(f"    - Training and evaluating: {name}")
    start_time = time.time()
    classifier.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Measure inference time on the entire test set
    start_inference_time = time.time()
    y_prob = classifier.predict_proba(X_test)[:, 1]
    inference_time_total = time.time() - start_inference_time
    
    y_pred = (y_prob >= 0.5).astype(int)
    
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    
    return {
        'Classifier': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'PR-AUC': auc_score(recall, precision),
        'Train-Time(s)': train_time,
        'Inference-Time-Total(s)': inference_time_total,
        'Test-Samples': len(X_test)
    }, classifier

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
    args = parser.parse_args()
    EMBED_DIR = args.embed_dir
    RESULTS_DIR = args.results_dir
    FIGURES_DIR = args.figures_dir
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_data(args.data)
    if df is None:
        return
    y = df["label"].values
    indices = np.arange(len(y))

    # Split data indices
    train_idx, test_idx = train_test_split(indices, test_size=args.test_size, stratify=y, random_state=SEED)
    print(f"\nSplitting data: {len(train_idx)} train, {len(test_idx)} test samples.")

    # Leakage guard: only pipeline 3's dataset has a doc_id column
    if "doc_id" in df.columns:
        train_idx, test_idx = resolve_leakage(df["doc_id"].values, train_idx, test_idx)
        print(f"   - Post-guard split: {len(train_idx)} train, {len(test_idx)} test samples.")

    y_train, y_test = y[train_idx], y[test_idx]

    # Discover available embeddings
    available_embeddings = {p.stem.replace('_prompt', ''): p for p in EMBED_DIR.glob('*_prompt.npy')}
    if not available_embeddings:
        print("❌ Error: No embedding files found in the 'embeddings' directory. Please run the generation script first.")
        return

    all_results = []
    classifiers = get_classifiers()

    # Process each embedding file
    for emb_name, emb_path in available_embeddings.items():
        print(f"\n{'='*70}\n🚀 Processing Embeddings: {emb_name.upper()}\n{'='*70}")
        
        embeddings = np.load(emb_path)
        X_train, X_test = embeddings[train_idx], embeddings[test_idx]
        
        plot_dimensionality_reduction(embeddings, y, emb_name, FIGURES_DIR)
        
        trained_models = {}
        for clf_name, classifier in classifiers.items():
            result, trained_model = train_evaluate(X_train, y_train, X_test, y_test, classifier, clf_name)
            result['Embeddings'] = emb_name.upper()
            all_results.append(result)
            trained_models[clf_name] = trained_model
        
        plot_roc_pr_curves(X_test, y_test, trained_models, emb_name, FIGURES_DIR)

    # --- Final Results Processing ---
    if not all_results:
        print("\n❌ No models were trained. Check for issues during the process.")
        return
        
    results_df = pd.DataFrame(all_results).sort_values(by='ROC-AUC', ascending=False).reset_index(drop=True)
    
    # Calculate and add per-sample inference time
    results_df['Inference-Time-per-Sample(ms)'] = (results_df['Inference-Time-Total(s)'] / results_df['Test-Samples']) * 1000
    
    print("\n\n" + "="*100 + "\n" + " " * 35 + "FINAL RESULTS SUMMARY" + "\n" + "="*100)
    display_cols = ['Embeddings', 'Classifier', 'Accuracy', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'Inference-Time-per-Sample(ms)']
    print(results_df[display_cols].to_string(index=False, float_format="%.4f"))
    print("\n" + "="*100)

    # Inference Time Summary
    avg_inference_ms = results_df['Inference-Time-per-Sample(ms)'].mean()
    print(f"\n⏱️  Inference Time Analysis:")
    print(f"   Our experiments show an average inference time of {avg_inference_ms:.2f}ms per sample.")
    
    # Display and save best configuration
    best = results_df.iloc[0]
    print(f"\n🏆 Best Configuration: {best['Embeddings']} + {best['Classifier']}")
    print(f"   - ROC-AUC: {best['ROC-AUC']:.4f}, Accuracy: {best['Accuracy']:.4f}, F1-Score: {best['F1-Score']:.4f}")
    print(f"   - Inference Time: {best['Inference-Time-per-Sample(ms)']:.2f}ms per sample")

    # Save results to files
    print("\n💾 Saving results...")
    results_df.to_csv(RESULTS_DIR / 'full_evaluation_results.csv', index=False)
    print(f"   ✔️ Full results saved to: {RESULTS_DIR / 'full_evaluation_results.csv'}")

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
        'best_configuration': best.to_dict(),
        'all_results': results_df.to_dict('records'),
        'info': {
            'test_size': args.test_size, 
            'seed': SEED,
            'average_inference_time_ms': avg_inference_ms
        }
    }
    with open(RESULTS_DIR / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"   ✔️ Summary saved to: {RESULTS_DIR / 'evaluation_summary.json'}")

    # Create final summary plots
    plot_performance_summary(results_df, FIGURES_DIR)
    print(f"   ✔️ Performance plots saved in: {FIGURES_DIR}/")
    
    print("\n🎉🎉🎉 Pipeline completed successfully! 🎉🎉🎉")

if __name__ == "__main__":
    main()
