"""Record the machine, package versions and file checksums behind a run, so its numbers can be checked later."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from pipeline_paths import DATA_DIR, RUNS_DIR  # noqa: E402

OUT_DIR = RUNS_DIR / "records"   # BUILD_RECORD.txt, pip_freeze.txt, SHA256SUMS
PACKAGES = ["scikit-learn", "xgboost", "lightgbm", "numpy", "joblib", "pandas", "scipy", "torch",
            "sentence-transformers", "transformers", "openai", "tiktoken", "datasets", "nltk",
            "presidio-analyzer", "psutil"]


def run(cmd: list[str]) -> str:
    """A command's output, or a note that it could not be run."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"(not available: {e})"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_record() -> str:
    lines = [f"recorded_at: {datetime.now().isoformat(timespec='seconds')}",
             f"platform: {platform.platform()}",
             f"python: {sys.version.split()[0]}"]
    try:
        import psutil
        import train_evaluate_visualize as tev   # the same machine facts the training runs record
        lines += [f"cpu: {tev._cpu_name()}",
                  f"cpu_cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical",
                  f"ram: {tev.ram_report()}"]
    except Exception as e:   # the record is still useful without these
        lines.append(f"cpu/ram: (not available: {e})")
    lines.append("gpu: " + run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                "--format=csv,noheader"]))
    try:
        import torch
        lines.append(f"torch_cuda: {torch.version.cuda}, available: {torch.cuda.is_available()}")
    except Exception as e:
        lines.append(f"torch_cuda: (not available: {e})")
    commit = run(["git", "-C", str(ROOT_DIR), "rev-parse", "HEAD"])
    lines.append("git_commit: " + (commit or "(not available)"))
    if commit:   # without a commit, an empty status would wrongly read as "nothing uncommitted"
        lines.append("git_uncommitted: " + (run(["git", "-C", str(ROOT_DIR), "status", "--short"]) or "none").replace("\n", "; "))
    lines.append("packages:")
    for name in PACKAGES:
        try:
            lines.append(f"  {name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            lines.append(f"  {name}: not installed")
    return "\n".join(lines) + "\n"


def checksum_targets() -> list[Path]:
    files = sorted(DATA_DIR.glob("*.jsonl")) + sorted((DATA_DIR / "attacks").glob("*.json"))
    files += sorted((ROOT_DIR / "src").glob("*.py")) + [ROOT_DIR / "pipeline_paths.py", ROOT_DIR / "requirements.txt"]
    # result files only: the embeddings are 8 GB and are covered by the results made from them
    files += sorted(RUNS_DIR.glob("pipeline*/*/results/*"))
    files += sorted(RUNS_DIR.glob("pipeline*/ablation_*.csv")) + sorted((RUNS_DIR / "analysis").glob("*"))
    return [f for f in files if f.is_file()]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "BUILD_RECORD.txt").write_text(build_record(), encoding="utf-8")
    (OUT_DIR / "pip_freeze.txt").write_text(run([sys.executable, "-m", "pip", "freeze"]) + "\n", encoding="utf-8")
    targets = checksum_targets()
    with (OUT_DIR / "SHA256SUMS").open("w", encoding="utf-8") as out:
        for f in targets:
            try:
                rel = f.relative_to(ROOT_DIR).as_posix()
            except ValueError:   # data or runs kept outside the repo (THESIS_DATA_DIR / THESIS_RUNS_DIR)
                rel = f.as_posix()
            out.write(f"{sha256(f)}  {rel}\n")
    print(f"Records written to {OUT_DIR}: BUILD_RECORD.txt, pip_freeze.txt, SHA256SUMS ({len(targets)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
