"""Where the dashboard reads data and runs from (pipeline_paths), and the Files tab's path prefixes."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # importable however the server was started
    sys.path.insert(0, str(ROOT))

from pipeline_paths import DATA_DIR, RUNS_DIR, SMOKE_SUBDIR  # noqa: E402

# Files tab paths look like "<prefix>/a/b", one prefix per tree
DATA_PREFIX, RUNS_PREFIX = "data", "runs"
INSPECT_ROOTS = {DATA_PREFIX: DATA_DIR, RUNS_PREFIX: RUNS_DIR}


def resolve(rel: str) -> Path | None:
    """Map a Files tab path onto disk, or None if it names an unknown tree or escapes its root."""
    parts = [p for p in rel.split("/") if p]
    if not parts or parts[0] not in INSPECT_ROOTS:
        return None
    base = INSPECT_ROOTS[parts[0]].resolve()
    p = base.joinpath(*parts[1:]).resolve()
    if p != base and base not in p.parents:
        return None
    return p


def write_run_dir(pipeline: str, mode: str, smoke: bool = False) -> Path:
    """Where a dashboard-launched run writes; smoke (row-limited) runs get their own subtree."""
    return RUNS_DIR / SMOKE_SUBDIR / pipeline / mode if smoke else RUNS_DIR / pipeline / mode


def describe() -> dict:
    """Which folders the dashboard reads, for /api/state and app.doctor."""
    return {"root": str(ROOT), "data_dir": str(DATA_DIR), "runs_dir": str(RUNS_DIR),
            "data_exists": DATA_DIR.is_dir(), "runs_exists": RUNS_DIR.is_dir()}
