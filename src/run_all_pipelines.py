"""Run every pipeline x ablation mode: embed (cached embeddings are reused unless --force), then train and evaluate."""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_paths import DATA_DIR, DATASETS, MODES, RUNS_DIR  # noqa: E402

import progress  # src/progress.py: the marker lines the dashboard's bars read


def describe_exit(code: int) -> str:
    """Human-readable exit status; a negative code means the child was killed by a signal."""
    if code >= 0:
        return f"exit {code}"
    import signal
    try:
        sig = signal.Signals(-code)
        return (f"exit {code} — killed by {sig.name} ({sig.value}); "
                f"look for the faulthandler dump above")
    except ValueError:
        return f"exit {code} — killed by signal {-code}"


def run_step(cmd: list[str], label: str) -> bool:
    """Run one stage; a failure is reported and skipped instead of aborting the sweep."""
    # flush before the child runs: a native crash would otherwise lose the buffered log lines
    progress.step(label)
    print(f"\n{'='*70}\n>>> {label}\n{'='*70}", flush=True)
    print("   $", " ".join(str(c) for c in cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n  FAILED ({describe_exit(result.returncode)}): {label}", flush=True)
        print("     continuing with the remaining runs.", flush=True)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="Recompute embeddings even if they are already on disk.")
    args = parser.parse_args()

    print(f"  datasets: {DATA_DIR}")
    print(f"  runs:     {RUNS_DIR}")

    failures: list[str] = []
    for pipeline, data_name in DATASETS.items():
        data_path = DATA_DIR / data_name
        if not data_path.is_file():
            print(f"Skipping {pipeline}: dataset not found at {data_path}")
            continue

        for mode in MODES:
            run_dir = RUNS_DIR / pipeline / mode
            gen_cmd = [sys.executable, "-u", str(SCRIPT_DIR / "generate_embeddings.py"),
                       "--data", str(data_path), "--mode", mode,
                       "--embed-dir", str(run_dir / "embeddings")]
            if args.force:
                gen_cmd.append("--force")
            if not run_step(gen_cmd, f"{pipeline} / {mode} — generate_embeddings"):
                failures.append(f"{pipeline}/{mode} — generate_embeddings")
                continue   # no embeddings -> nothing to train on

            eval_cmd = [sys.executable, "-u", str(SCRIPT_DIR / "train_evaluate_visualize.py"),
                        "--data", str(data_path), "--run-dir", str(run_dir)]
            if not run_step(eval_cmd, f"{pipeline} / {mode} — train_evaluate_visualize"):
                failures.append(f"{pipeline}/{mode} — train_evaluate_visualize")

    if failures:
        print(f"\n{'='*70}\n{len(failures)} step(s) FAILED — everything else completed:")
        for f in failures:
            print(f"     · {f}")
        print("   Re-run to retry: finished embeddings are cached, so only the\n"
              "   missing ones are recomputed.\n" + "=" * 70)
        return 1

    print("\nAll pipelines completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
