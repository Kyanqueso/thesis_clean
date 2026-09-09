"""Download the Alamsabi BIPIA+GPT dataset into this machine's data tree."""

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline_paths import DATA_DIR, DATASETS  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out-dir", type=Path, default=DATA_DIR,
                    help="Directory to write the dataset into.")
args = parser.parse_args()

from datasets import load_dataset  # heavy, and not needed for --help

out_path = args.out_dir / DATASETS["pipeline1_alamsabi"]
out_path.parent.mkdir(parents=True, exist_ok=True)

dataset = load_dataset("MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT")

with open(out_path, "w", encoding="utf-8") as f:
    for item in dataset["train"]:
        json.dump(item, f)
        f.write("\n")

print(f"Dataset downloaded successfully to {out_path}")
