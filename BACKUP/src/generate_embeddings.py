import argparse
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken
import torch
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import progress  # src/progress.py: the marker lines the dashboard's bars read

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
# normal mode embeds user_intent + SEPARATOR + context; app/detect.py must build it the same way
SEPARATOR = "\n\n---\n\n"

try:
    client = OpenAI() # Reads OPENAI_API_KEY from environment variables automatically
    OPENAI_ENABLED = True
except Exception:
    client = None
    OPENAI_ENABLED = False
    print("Warning: OpenAI API key not found or invalid. OpenAI models will be skipped.")

# OpenAI is limited by tokens per minute; set OPENAI_TPM to your account's real limit
OPENAI_TPM = int(os.environ.get("OPENAI_TPM", 1_000_000))
OPENAI_TPM_TARGET = float(os.environ.get("OPENAI_TPM_TARGET", 0.85))  # headroom
OPENAI_REQUEST_TOKENS = 100_000   # tokens per request; well under the 300k cap
OPENAI_MAX_TOKENS = 8191          # per-input limit for the model


class _TokenBudget:
    """Rolling 60s token accountant. take(n) blocks until n tokens fit."""

    def __init__(self, limit: int, target: float):
        self.cap = max(1, int(limit * target))
        self.events: deque = deque()   # (timestamp, tokens)
        self.used = 0
        self.waited = 0.0

    def _expire(self, now: float):
        while self.events and now - self.events[0][0] >= 60.0:
            self.used -= self.events.popleft()[1]

    def take(self, tokens: int):
        while True:
            now = time.monotonic()
            self._expire(now)
            # `not self.events` lets an oversized batch through: nothing to wait for
            if self.used + tokens <= self.cap or not self.events:
                self.events.append((now, tokens))
                self.used += tokens
                return
            wait = 60.0 - (now - self.events[0][0]) + 0.05
            self.waited += wait
            time.sleep(max(wait, 0.05))


def _retry_after(exc) -> float | None:
    """Seconds OpenAI asked us to wait, if it said so."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    for key, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
        raw = headers.get(key)
        if raw:
            try:
                return float(raw) * scale + 0.1
            except (TypeError, ValueError):
                pass
    return None


def _token_batches(counts: list[int], budget: int, max_texts: int) -> list[tuple[int, int]]:
    """Contiguous [start, end) spans respecting both a token budget and a text cap."""
    spans, start, total = [], 0, 0
    for i, n in enumerate(counts):
        if i > start and (total + n > budget or i - start >= max_texts):
            spans.append((start, i))
            start, total = i, 0
        total += n
    spans.append((start, len(counts)))
    return spans


def _embed_batch(batch: list[str], model_id: str, attempts: int = 6):
    """One embeddings call, retried with backoff. Raises only if every try fails."""
    for k in range(attempts):
        try:
            return client.embeddings.create(model=model_id, input=batch)
        except Exception as e:  # rate limits, timeouts, transient 5xx
            if k == attempts - 1:
                raise
            wait = _retry_after(e) or min(2 ** k, 30)
            print(f"\n  {type(e).__name__} — retry {k + 1}/{attempts - 1} in {wait:.1f}s")
            time.sleep(wait)


class _GpuUse:
    """Peak VRAM and whole-card GPU utilisation over a block; utilisation needs nvidia-ml-py."""

    POLL_S = 0.05

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.util, self._stop, self._thread = [], threading.Event(), None
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._thread = threading.Thread(target=self._poll, args=(pynvml, handle), daemon=True)
            self._thread.start()
        except Exception as e:   # no NVML: VRAM is still measured
            print(f"    GPU utilisation not measured ({type(e).__name__}: {e})")
        self._t0 = time.perf_counter()
        return self

    def _poll(self, nvml, handle):
        while not self._stop.is_set():
            self.util.append(nvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            time.sleep(self.POLL_S)

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self._stop.set()
        if self._thread:
            self._thread.join()
        self.result = {
            "device": torch.cuda.get_device_name(0),
            "embed_s": time.perf_counter() - self._t0,
            "vram_peak_mb": torch.cuda.max_memory_reserved() / 1e6,
            "vram_alloc_peak_mb": torch.cuda.max_memory_allocated() / 1e6,
            "gpu_util_mean": float(np.mean(self.util)) if self.util else None,
            "gpu_util_max": float(np.max(self.util)) if self.util else None,
        }
        return False


def generate_embeddings(texts: list[str], model_id: str, batch_size: int, embed_type: str, dtype: str = None, max_seq_length: int = None, item: str = "", gpu_use: dict | None = None) -> np.ndarray:
    """gpu_use, when given, is filled with what embedding on a CUDA GPU cost (left empty otherwise)."""

    if embed_type == 'sbert':
        progress.mark("load", item)
        model_kwargs = {"torch_dtype": getattr(torch, dtype)} if dtype else None
        model = SentenceTransformer(model_id, model_kwargs=model_kwargs)
        if max_seq_length:
            model.max_seq_length = max_seq_length
        progress.mark("embed", item)   # no total: the dashboard shows it as "working"
        meter = _GpuUse() if model.device.type == "cuda" else contextlib.nullcontext()
        with meter:
            out = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=False
            )
        if gpu_use is not None and isinstance(meter, _GpuUse):
            gpu_use.update(meter.result, n_texts=len(texts), batch_size=batch_size)
        return out

    if embed_type == 'openai':
        if not OPENAI_ENABLED:
            raise RuntimeError("OpenAI API key is not configured. Cannot generate embeddings.")

        tokenizer = tiktoken.encoding_for_model(model_id)

        # tokenize once: the counts drive the pacing, and truncation keeps every input under the limit
        prepared, counts, n_truncated = [], [], 0
        for t in progress.track(tqdm(texts, desc="Tokenizing", leave=False),
                                "tokenize", item, total=len(texts)):
            t = t if t and t.strip() else " "   # the API rejects empty strings
            ids = tokenizer.encode(t)
            if len(ids) > OPENAI_MAX_TOKENS:
                ids = ids[:OPENAI_MAX_TOKENS]
                t = tokenizer.decode(ids)
                n_truncated += 1
            prepared.append(t)
            counts.append(len(ids))

        spans = _token_batches(counts, OPENAI_REQUEST_TOKENS, batch_size)
        total_tokens = sum(counts)
        budget = _TokenBudget(OPENAI_TPM, OPENAI_TPM_TARGET)
        floor_min = total_tokens / budget.cap
        print(f"    {total_tokens:,} tokens in {len(spans):,} requests "
              f"({n_truncated:,} truncated) · pacing at {budget.cap:,} tokens/min "
              f"→ ~{floor_min:.1f} min minimum")

        all_embeddings = []
        for a, b in progress.track(tqdm(spans, desc=f"OpenAI ({model_id})"),
                                   "embed", item, total=len(spans)):
            budget.take(sum(counts[a:b]))          # wait if we would exceed the cap
            response = _embed_batch(prepared[a:b], model_id)
            all_embeddings.extend(r.embedding for r in response.data)

        if budget.waited > 1:
            print(f"    (paused {budget.waited:.0f}s total to stay under the rate limit)")
        return np.array(all_embeddings, dtype=np.float32)

    raise ValueError(f"Unsupported embedding type: {embed_type}")

MODELS = {
    "openai": {
        "model_id": "text-embedding-3-small",
        "batch_size": 256,  # texts per request; OPENAI_REQUEST_TOKENS is the real limit
        "embed_type": "openai",
        "enabled": OPENAI_ENABLED,
        "description": "OpenAI text-embedding-3-small (1536 dims)"
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-Embedding-4B",
        "batch_size": 4,
        "embed_type": "sbert",
        "dtype": "float16",
        "max_seq_length": 2048,
        "enabled": True,
        "description": "Qwen3-Embedding-4B (2560 dims)"
    },
    "minilm": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "batch_size": 128,
        "embed_type": "sbert",
        "enabled": True,
        "description": "MiniLM-L6-v2 (384 dims)"
    }
}

def main():
    parser = argparse.ArgumentParser(description="Embed one dataset with each embedding model.")
    parser.add_argument("--data", type=Path, required=True, help="The dataset (.jsonl).")
    parser.add_argument("--embed-dir", type=Path, required=True, help="A run's embeddings/ folder.")
    parser.add_argument("--mode", choices=["normal", "user_intent_only", "context_only"], default="normal",
                        help="Embed intent + context, the intent alone, or the context alone.")
    parser.add_argument("--models", help=f"Comma-separated subset of {', '.join(MODELS)}. Default: all.")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings that already exist.")
    parser.add_argument("--_single", help=argparse.SUPPRESS)
    args = parser.parse_args()
    EMBED_DIR = args.embed_dir
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    # what the caller asked for; "enabled" can also go False on its own (e.g. no OpenAI key)
    requested = sorted(MODELS.keys())
    if args.models:
        selected = {m.strip() for m in args.models.split(",")}
        unknown = selected - set(MODELS.keys())
        if unknown:
            print(f"Error: Unknown model(s) in --models: {', '.join(unknown)}. Available: {', '.join(MODELS.keys())}")
            return 1
        requested = sorted(selected)
        for key in MODELS:
            if key not in selected:
                MODELS[key]["enabled"] = False

    # each model runs in its own subprocess, so a CUDA crash cannot break the next one
    if args._single is None:
        print("\nStarting embedding generation (each model runs in its own subprocess)...")
        for key, config in MODELS.items():
            if not config["enabled"]:
                print(f"\nSkipping '{key}' model (disabled).")
                continue

            output_path = EMBED_DIR / f"{key}_prompt.npy"
            if output_path.exists() and not args.force:
                print(f"\n--- {key.upper()}: embeddings already exist. Use --force to overwrite. ---")
                continue

            # the GPU record belongs to the old embeddings; the worker writes a new one if it measures
            (EMBED_DIR / f"{key}_gpu.json").unlink(missing_ok=True)
            print(f"\n--- Launching subprocess for: {key.upper()} ---")
            cmd = [
                sys.executable, str(Path(__file__).resolve()),
                "--data", str(args.data),
                "--mode", args.mode,
                "--embed-dir", str(EMBED_DIR),
                "--_single", key,
            ]
            cwd = None
            # qwen3 can run on the Vast.ai instance from Settings instead (see app/vast.py)
            if key == "qwen3" and os.environ.get("THESIS_QWEN3_REMOTE", "") not in ("", "0"):
                print("    (running on the Vast.ai instance)")
                cmd = [
                    sys.executable, "-u", "-m", "app.vast", "embed",
                    "--data", str(args.data.resolve()),
                    "--mode", args.mode,
                    "--embed-dir", str(EMBED_DIR.resolve()),
                ]
                cwd = ROOT_DIR
            if args.force:
                cmd.append("--force")
            result = subprocess.run(cmd, cwd=cwd)
            if result.returncode != 0:
                print(f"  Subprocess for '{key}' exited with code {result.returncode}.")

        # fail when a requested backend produced nothing, so the sweep records it as a failure
        absent = [k for k in requested if not (EMBED_DIR / f"{k}_prompt.npy").exists()]
        print("\nEmbedding files: " + ", ".join(f"{k} {'MISSING' if k in absent else 'ok'}" for k in requested))
        if absent:
            print(f"No embeddings produced for: {', '.join(absent)} "
                  f"(missing OPENAI_API_KEY? or pick backends with --models)")
            return 1
        print("\nEmbedding generation completed!")
        return 0

    if not args.data.is_file():
        print(f"Error: Data file not found at: {args.data}")
        return 1

    print(f"Loading data from: {args.data.name}...")
    try:
        df = pd.read_json(args.data, lines=True)
    except ValueError as e:
        print(f"Error reading JSONL file: {e}")
        return 1

    print(f"   {len(df):,} samples" + (f", {df['label'].mean():.1%} malicious" if "label" in df else ""))

    if args.mode == "user_intent_only":
        print("\nMode: user_intent_only — embedding user_intent alone")
        combined_texts = df["user_intent"].astype(str).tolist()
    elif args.mode == "context_only":
        print("\nMode: context_only — embedding context alone")
        combined_texts = df["context"].astype(str).tolist()
    else:
        print(f"\nMode: normal — user intent first, then context, joined by: '{SEPARATOR.strip()}'")
        # intent first, so a truncating model (MiniLM: 256 tokens) loses the end of the context, never the intent
        combined_texts = [
            f"{intent}{SEPARATOR}{context}"
            for intent, context in zip(df["user_intent"].astype(str), df["context"].astype(str))
        ]

    print("Sample combined text (first 200 chars):")
    print(f"   '{combined_texts[0][:200]}...'")

    # worker: embed the one model the orchestrator launched it for
    key = args._single
    config = MODELS[key]
    print(f"\n--- Processing: {key.upper()} ---")
    print(f"    Model: {config['description']}")
    gpu_use = {}
    try:
        embeddings = generate_embeddings(
            texts=combined_texts,
            model_id=config["model_id"],
            batch_size=config["batch_size"],
            embed_type=config["embed_type"],
            dtype=config.get("dtype"),
            max_seq_length=config.get("max_seq_length"),
            item=key,
            gpu_use=gpu_use,
        )
    except Exception as e:
        print(f"  Error generating embeddings for '{key}': {e}")
        return 1

    output_path = EMBED_DIR / f"{key}_prompt.npy"
    np.save(output_path, embeddings)
    print(f"  Saved embeddings to: {output_path.name}")
    if gpu_use:   # read by the dashboard's Resources table
        gpu_use["measured_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        (EMBED_DIR / f"{key}_gpu.json").write_text(json.dumps(gpu_use, indent=2), encoding="utf-8")
        print(f"      GPU: {gpu_use['vram_peak_mb']:.0f} MB VRAM peak"
              + (f", {gpu_use['gpu_util_mean']:.0f}% average use" if gpu_use["gpu_util_mean"] is not None else ""))
    print(f"      Shape: {embeddings.shape}")
    print(f"      Size: {embeddings.nbytes / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
