# Indirect Prompt Injection Detection (Thesis)

Detects indirect prompt injection by embedding `(context, user_intent)` pairs and
classifying them with a decision threshold. A local dashboard shows the results
and lets you test the trained classifiers on your own text.

These steps were tested on a fresh clone on Windows 11 with Python 3.13.
macOS and Linux should work the same way (the commands for them are listed too),
but nobody on the team has tried them yet.

---

## What you need

- **Python 3.13** from [python.org](https://www.python.org/downloads/).
  On Windows, tick **"Add python.exe to PATH"** in the installer.
- **Git**
- **About 12 GB free disk space**: runs 7.6 GB, data 0.4 GB, Python packages about 2 GB, models 0.25 GB
- **16 GB RAM** is enough. **No GPU needed.**
- **Internet**, the first time only, to install packages and download the small MiniLM model.

## 1. Get the code

```powershell
git clone https://github.com/Kyanqueso/thesis_clean.git
cd thesis_clean
```

## 2. Install the Python packages

Use a virtual environment so these exact versions don't clash with other projects.

**Windows (PowerShell):**

```powershell
py -3.13 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux:**

```bash
python3.13 -m venv venv
venv/bin/python -m pip install --upgrade pip
venv/bin/python -m pip install -r requirements.txt
```

The install takes about 5–10 minutes. The commands call the venv's Python directly,
so you don't need to activate it. That also avoids PowerShell's "running scripts is
disabled" error.

Keep the versions in `requirements.txt` exactly as they are. The shared models were
saved with scikit-learn 1.7.1, xgboost 3.0.4, lightgbm 4.7.0, numpy 2.2.6 and
joblib 1.5.3, and they may fail to load with other versions.

## 3. Get the data, runs and models (Google Drive)

`data/` and `runs/` are too big for git (they're gitignored), so they're on the team
Drive: **[Drive link]**

Download the three files and extract them **into the `thesis_clean` folder**, so it
looks like this:

```
thesis_clean/
├── app/
├── src/
├── data/
│   ├── final_training_dataset.jsonl
│   ├── organic_bipia_dataset.jsonl
│   └── organic_injected_dataset.jsonl
├── runs/
│   ├── all_pipelines_summary.csv
│   ├── pipeline1_alamsabi/
│   │   ├── normal/
│   │   │   ├── embeddings/
│   │   │   ├── figures/
│   │   │   ├── results/
│   │   │   └── models/        ← from thesis_models_only.zip
│   │   ├── user_intent_only/
│   │   └── context_only/
│   ├── pipeline2_organic_bipia/
│   └── pipeline3_organic_injected/
└── requirements.txt
```

1. **`data` zip:** this gives you `thesis_clean/data/`.
2. **`runs` zip:** this gives you `thesis_clean/runs/`.
3. **`thesis_models_only.zip`:** extract it into `thesis_clean` itself. It already
   contains `runs/pipeline…/…/models/` paths, so it adds a `models/` folder to each
   of the 9 runs and doesn't overwrite anything.

**Watch out for Windows "Extract All".** By default it puts everything inside an
extra folder named after the zip, for example
`thesis_clean/thesis_models_only/runs/...`. Change the destination to the
`thesis_clean` folder itself, or move the folders up afterwards.

## 4. Check that everything was found

```powershell
venv\Scripts\python.exe -m app.doctor
```

On macOS/Linux, use `venv/bin/python` instead. You should see:

- **Datasets:** all 3 marked `ok`.
- **Runs:** each of the 9 lines looks like `E:OQM R:y P:y F:10 M:15` (all three
  embeddings, results, predictions, 9–10 figures and 15 models).
- **Totals:** `runs with results: 9/9 · with predictions: 9/9 · with models: 9/9`.

If something shows as missing, the folder is almost always one level too deep
(see the warning in step 3).

## 5. Start the dashboard

```powershell
venv\Scripts\python.exe -m uvicorn app.server:app --port 8756
```

Open **http://127.0.0.1:8756** in your browser. Stop the server with `Ctrl+C`.

Don't add `--reload` while a run from the Run tab is training. `--reload` restarts
the server whenever a `.py` file changes, and that kills the running job.

## 6. Use the Simulation tab

Pick a pipeline, mode, embedding and classifier, then type a user intent and a
context to get a malicious/benign verdict.

| Embedding | Works out of the box? | Notes |
|---|---|---|
| **minilm** | ✅ Yes | Runs locally. The first request takes about 30 s because it downloads the model once; after that each request takes milliseconds. |
| **openai** | Needs a key | The text you type is embedded through OpenAI's API. Create a file named `.env` in `thesis_clean` containing `OPENAI_API_KEY=sk-...`, then restart the server. Without a key you get a "Missing credentials" error. |
| **qwen3** | ❌ Not practical on laptops | It loads a 4-billion-parameter model (about 8 GB). |

The models were trained on the embeddings that are already in `runs/`, so testing
them never needs a GPU or recomputed embeddings.

---

## Retraining (optional, you normally don't need this)

`models/` already has all 135 classifiers (9 runs × 3 embeddings × 5 classifiers).
If you do want to retrain, use the dashboard's **Run** tab:

- **Settings:** keep **save models** ticked, tick **skip projections** (the t-SNE/UMAP
  charts already exist), and don't tick **force re-embed**, which would recompute the
  embeddings.
- **Time:** the full sweep takes about **5–8 hours** on a 24-thread laptop.
  `pipeline1_alamsabi/user_intent_only` alone is about 3 hours.
- **Windows Update:** pause it first (Settings → Windows Update → Pause), or it may
  reboot mid-run. Keep the laptop plugged in and stop it from sleeping.
- **Your results change:** retraining overwrites that run's `results/` and `figures/`.
  Accuracy numbers stay the same (the seed is fixed at 42), but latency and
  training-time numbers become your laptop's. The thesis uses the Drive copy.
- **Command line:** without the dashboard, run
  `venv\Scripts\python.exe src\run_all_pipelines.py --save-models --skip-projections`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip install -r requirements.txt` fails on `nvidia-cufile-cu12` | You have an old copy of `requirements.txt`. Run `git pull`. |
| `No module named 'sentence_transformers'` (or `tiktoken`, `lightgbm`) in a job log | The packages went into a different Python. Start the server with `venv\Scripts\python.exe` as in step 5. |
| `No module named '_posixsubprocess'` | joblib is older than 1.5.3. Reinstall from `requirements.txt`. |
| A run crashes with `Tcl_AsyncDelete: async handler deleted by the wrong thread` | Old code. Run `git pull` (plots now use matplotlib's `Agg` backend). |
| Simulation says "No trained classifiers yet" | `models/` isn't inside the run folders. Run `app.doctor` and check `M:15`. |
| Datasets show as missing even though you copied them | Check the folder layout in step 3, then restart the server. It picks its folders once, at startup. |
| Error "address already in use" on port 8756 | A server is already running. Close it, or use `--port 8757` and open that port instead. |
