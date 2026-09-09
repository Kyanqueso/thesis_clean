"""Single definition of where this project's dataset and run trees live.

Machines name them differently: this checkout has data/ + runs/ (runs/ being a
junction into the supervisor's delivery), the main machine has data_from_vastai/
+ runs_from_vastai/. The dashboard (app/paths.py) and the pipeline scripts in
src/ both locate them through here, so a checkout dropped onto a new machine
finds its artifacts without edits.

A tree is identified by its *content*, not its name — a runs tree holds
<pipeline>/<mode>/ directories, a dataset tree holds the pipeline .jsonl files —
so an unexpected name still works. Order, per tree:

  1. the env override, if set (absolute, or relative to the repo root):
       THESIS_DATA_DIR   THESIS_RUNS_DIR   THESIS_RUNS_WRITE_DIR
  2. a directory named like the tree that holds the right content
  3. any directory near the repo root holding the right content: the root, its
     subdirectories two deep, then the root's siblings (for a dashboard checked
     out beside the artifacts instead of inside them)
  4. a directory named like the tree, even if empty
  5. the plain name, created on demand by whatever writes into it

`python -m app.doctor` prints what this resolved to and how.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_CANDIDATES = ("data_from_vastai", "data")
RUNS_CANDIDATES = ("runs_from_vastai", "runs")

DATASETS = {
    "pipeline1_alamsabi": "final_training_dataset.jsonl",
    "pipeline2_organic_bipia": "organic_bipia_dataset.jsonl",
    "pipeline3_organic_injected": "organic_injected_dataset.jsonl",
}
MODES = ["normal", "user_intent_only", "context_only"]

SMOKE_SUBDIR = "_smoke"
SUMMARY_FILES = ("all_pipelines_summary.csv", "all_pipelines_summary.html")

# never worth walking into while looking for artifact trees
SKIP_NAMES = {
    "venv", ".venv", "env", "node_modules", "__pycache__", "site-packages",
    "Lib", "Scripts", "Include", "app", "src", ".git", ".idea", ".vscode",
    ".claude", ".ipynb_checkpoints",
}


def env_dir(var: str) -> Path | None:
    raw = os.environ.get(var, "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def is_runs_tree(d: Path) -> bool:
    """A runs tree has at least one <pipeline>/<mode>/ directory."""
    return any((d / pipeline / mode).is_dir() for pipeline in DATASETS for mode in MODES)


def is_data_tree(d: Path) -> bool:
    """A dataset tree holds at least one of the pipeline .jsonl files."""
    return any((d / name).is_file() for name in DATASETS.values())


def _subdirs(d: Path) -> list[Path]:
    try:
        return [p for p in sorted(d.iterdir())
                if p.is_dir() and not p.name.startswith(".") and p.name not in SKIP_NAMES]
    except OSError:  # unreadable / disconnected drive
        return []


def probe_dirs(depth: int = 2) -> list[Path]:
    """Directories worth checking for artifacts, nearest first."""
    out: list[Path] = [ROOT]
    seen = {str(ROOT).lower()}

    def walk(start: Path, left: int):
        for d in _subdirs(start):
            key = str(d).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
            if left > 1:
                walk(d, left - 1)

    walk(ROOT, depth)
    if ROOT.parent != ROOT:  # dashboard checked out beside the artifacts
        walk(ROOT.parent, depth)
    return out


def resolve_tree(var: str, candidates: tuple[str, ...], looks_right) -> tuple[Path, str]:
    """Locate one tree. Returns (path, how it was found) — see module docstring."""
    override = env_dir(var)
    if override is not None:
        return override, f"{var} env override"

    named = [ROOT / name for name in candidates if (ROOT / name).is_dir()]
    for d in named:
        if looks_right(d):
            return d, "expected name, with artifacts in it"
    for d in probe_dirs():
        if looks_right(d):
            return d, "found by content"
    if named:
        return named[0], "expected name, currently empty"
    return ROOT / candidates[-1], "default name (nothing found yet)"


DATA_DIR, DATA_SOURCE = resolve_tree("THESIS_DATA_DIR", DATA_CANDIDATES, is_data_tree)
RUNS_DIR, RUNS_SOURCE = resolve_tree("THESIS_RUNS_DIR", RUNS_CANDIDATES, is_runs_tree)

# runs launched from the dashboard write here: the tree we read from, so there
# is one place to look. Override only to keep delivered artifacts pristine.
RUNS_WRITE_DIR = env_dir("THESIS_RUNS_WRITE_DIR") or RUNS_DIR


def summary_path(name: str) -> Path | None:
    """The sweep aggregate sits in the runs tree, but a run started from the
    repo root leaves it there instead; accept either."""
    for base in (RUNS_DIR, ROOT):
        p = base / name
        if p.is_file():
            return p
    return None
