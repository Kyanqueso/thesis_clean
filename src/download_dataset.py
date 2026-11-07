"""Download the Alamsabi BIPIA+GPT dataset (pipeline 1) into the data folder."""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from datasets import load_dataset  # noqa: E402

from pipeline_paths import DATA_DIR, DATASETS  # noqa: E402

import progress  # src/progress.py: the marker lines the dashboard's bars read

out_path = DATA_DIR / DATASETS["pipeline1_alamsabi"]
out_path.parent.mkdir(parents=True, exist_ok=True)

progress.mark("download", "Alamsabi BIPIA+GPT")
dataset = load_dataset("MAlmasabi/Indirect-Prompt-Injection-BIPIA-GPT")

with open(out_path, "w", encoding="utf-8") as f:
    for item in progress.track(dataset["train"], "write", out_path.name):
        json.dump(item, f)
        f.write("\n")

print(f"Dataset downloaded successfully to {out_path}")
