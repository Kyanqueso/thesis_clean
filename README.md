# Indirect Prompt Injection Detection (Thesis)

Detects indirect prompt injection by embedding `(context, user_intent)` pairs and classifying them with decision threshold.

## Setup (Via python venv)

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."
```

## Run

```bash
python src/download_dataset.py
python src/organic_dataset.py
python src/run_all_pipelines.py
```

## Output

Results, figures, and the summary HTML land in `runs/` folder.
