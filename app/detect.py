"""Live detection: embed one (user_intent, context) pair and score it with a saved model."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from app import paths
from app.runs import EMB_MODELS, MODES, PIPELINES, run_dir

SEPARATOR = "\n\n---\n\n"  # must match generate_embeddings.py default

_model_cache: dict[str, object] = {}
_embedder_cache: dict[str, object] = {}


def available_models() -> list[dict]:
    """Scan runs/*/*/models/*.joblib for scorable configurations."""
    out = []
    paths.ensure_current()
    if not paths.RUNS_DIR.is_dir():
        return out
    for pipeline in PIPELINES:
        for mode in MODES:
            models_dir = run_dir(pipeline, mode) / "models"
            if not models_dir.is_dir():
                continue
            for p in sorted(models_dir.glob("*.joblib")):
                emb, _, clf = p.stem.partition("_")
                if emb not in EMB_MODELS:
                    continue
                out.append({"pipeline": pipeline, "mode": mode, "emb": emb, "clf": clf,
                            "path": str(p)})
    return out


def options() -> dict:
    """Scorable models plus, when there are none, why and what fixes it."""
    models = available_models()
    if models:
        return {"models": models}
    return {
        "models": [],
        "reason": ("no saved classifiers in this tree — delivered runs ship results "
                   "and figures, but no models/ directory"),
        "fix": ("queue a run from the Run tab with “save models” checked (it is on "
                "by default); the models land in <run>/models/*.joblib"),
    }


def _get_embedder(emb: str):
    if emb in _embedder_cache:
        return _embedder_cache[emb]
    if emb == "minilm":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    elif emb == "qwen3":
        import torch
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("Qwen/Qwen3-Embedding-4B",
                                    model_kwargs={"torch_dtype": torch.float16})
        model.max_seq_length = 2048
    elif emb == "openai":
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv()
        model = OpenAI()  # raises if no key configured
    else:
        raise ValueError(f"unknown embedding model: {emb}")
    _embedder_cache[emb] = model
    return model


def _embed(emb: str, text: str) -> np.ndarray:
    embedder = _get_embedder(emb)
    if emb == "openai":
        resp = embedder.embeddings.create(model="text-embedding-3-small", input=[text])
        return np.array(resp.data[0].embedding, dtype=np.float32)
    return embedder.encode([text], normalize_embeddings=False)[0]


def score(pipeline: str, mode: str, emb: str, clf: str,
          user_intent: str, context: str) -> dict:
    import joblib

    model_path = run_dir(pipeline, mode) / "models" / f"{emb}_{clf}.joblib"
    if not model_path.is_file():
        return {"error": f"no saved model at {model_path.name} for {pipeline}/{mode}"}

    if mode == "user_intent_only":
        text = user_intent
    elif mode == "context_only":
        text = context
    else:
        text = f"{context}{SEPARATOR}{user_intent}"
    if not text.strip():
        return {"error": "empty input"}

    key = str(model_path)
    if key not in _model_cache:
        _model_cache[key] = joblib.load(model_path)
    classifier = _model_cache[key]

    t0 = time.perf_counter()
    vec = _embed(emb, text)
    embed_ms = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    prob = float(classifier.predict_proba(vec.reshape(1, -1))[0, 1])
    clf_ms = (time.perf_counter() - t1) * 1000

    return {
        "probability": prob,
        "verdict": "malicious" if prob >= 0.5 else "benign",
        "embed_ms": round(embed_ms, 2),
        "classify_ms": round(clf_ms, 3),
        "mode": mode, "emb": emb, "clf": clf, "pipeline": pipeline,
    }
