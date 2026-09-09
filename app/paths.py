"""Dashboard view of the project's trees, on top of the shared resolver.

pipeline_paths.py (repo root) decides *where* the dataset and run trees are on
this machine; this module adds what only the dashboard needs: the single-segment
prefixes the file explorer addresses paths by, and the write-side run dirs.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # importable however the server was started
    sys.path.insert(0, str(ROOT))

from pipeline_paths import (  # noqa: E402
    DATA_DIR, DATA_SOURCE, RUNS_DIR, RUNS_SOURCE, RUNS_WRITE_DIR,
    SMOKE_SUBDIR, SUMMARY_FILES, summary_path,
)

# Path prefixes the UI shows and the file explorer accepts. Single segments, so
# they stay stable when a tree lives outside the repo root.
DATA_PREFIX = DATA_DIR.name
RUNS_PREFIX = RUNS_DIR.name
if RUNS_PREFIX == DATA_PREFIX:  # pathological config; keep the two addressable
    DATA_PREFIX, RUNS_PREFIX = "data", "runs"

# Scripts run with their default dirs write to the repo root instead of a run
# dir; those legacy outputs stay browsable.
INSPECT_ROOTS: dict[str, Path] = {
    DATA_PREFIX: DATA_DIR,
    RUNS_PREFIX: RUNS_DIR,
    "figures": ROOT / "figures",
    "results": ROOT / "results",
    "embeddings": ROOT / "embeddings",
}
WRITE_PREFIX = RUNS_PREFIX
if RUNS_WRITE_DIR != RUNS_DIR:
    WRITE_PREFIX = RUNS_WRITE_DIR.name
    INSPECT_ROOTS.setdefault(WRITE_PREFIX, RUNS_WRITE_DIR)

# the sweep aggregate can sit at the repo root instead of in the runs tree; that
# one prefix exposes those two files only, never the rest of the root
ROOT_PREFIX = "root"
INSPECT_ROOTS[ROOT_PREFIX] = ROOT


def summary_rel(name: str) -> str:
    """Explorer path for a sweep summary file, wherever it actually lives."""
    found = summary_path(name)
    if found is not None and found.parent == ROOT:
        return f"{ROOT_PREFIX}/{name}"
    return f"{RUNS_PREFIX}/{name}"


def resolve(rel: str) -> Path | None:
    """Map a dashboard-relative path ('<prefix>/a/b') onto disk, or None if it
    escapes its root or names an unknown one."""
    parts = [p for p in rel.split("/") if p]
    if not parts or parts[0] not in INSPECT_ROOTS:
        return None
    if parts[0] == ROOT_PREFIX and (len(parts) != 2 or parts[1] not in SUMMARY_FILES):
        return None  # the root prefix is not a general file browser
    base = INSPECT_ROOTS[parts[0]].resolve()
    p = base.joinpath(*parts[1:]).resolve()
    if p != base and base not in p.parents:
        return None
    return p


def write_run_dir(pipeline: str, mode: str, smoke: bool = False) -> Path:
    """Where a dashboard-launched run writes. Smoke runs (--limit) get their own
    subtree so a toy evaluation can never overwrite a delivered one."""
    if smoke:
        return RUNS_WRITE_DIR / SMOKE_SUBDIR / pipeline / mode
    return RUNS_WRITE_DIR / pipeline / mode


def describe() -> dict:
    """What the UI shows, so it is obvious which tree is being read."""
    return {
        "root": str(ROOT),
        "data_dir": str(DATA_DIR),
        "runs_dir": str(RUNS_DIR),
        "runs_write_dir": str(RUNS_WRITE_DIR),
        "data_prefix": DATA_PREFIX,
        "runs_prefix": RUNS_PREFIX,
        "data_exists": DATA_DIR.is_dir(),
        "runs_exists": RUNS_DIR.is_dir(),
        "data_source": DATA_SOURCE,
        "runs_source": RUNS_SOURCE,
        "split_write": RUNS_WRITE_DIR != RUNS_DIR,
    }
