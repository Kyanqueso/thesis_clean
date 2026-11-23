"""Where the datasets and run outputs live: data/ and runs/ in the repo root, or THESIS_DATA_DIR / THESIS_RUNS_DIR."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASETS = {
    "pipeline1_alamsabi": "final_training_dataset.jsonl",
    "pipeline2_organic_bipia": "organic_bipia_dataset.jsonl",
    "pipeline3_organic_injected": "organic_injected_dataset.jsonl",
}
MODES = ["normal", "user_intent_only", "context_only"]

SMOKE_SUBDIR = "_smoke"


def env_dir(var: str, default: str) -> Path:
    """The directory named by an env var (absolute, or relative to the repo root), else ROOT/default."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        return ROOT / default
    p = Path(raw).expanduser()
    return p if p.is_absolute() else ROOT / p


DATA_DIR = env_dir("THESIS_DATA_DIR", "data")
RUNS_DIR = env_dir("THESIS_RUNS_DIR", "runs")
