#!/usr/bin/env python
"""
Embedding Generation Script
===========================
This script generates embeddings by combining user intent and external content
into a single text.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

# Attempt to import required libraries and provide guidance on failure.
try:
    from sentence_transformers import SentenceTransformer
    from openai import OpenAI
    from dotenv import load_dotenv
    import tiktoken
except ImportError:
    print("❌ Error: Required libraries are not installed.")
    print("   Please run: pip install sentence-transformers openai python-dotenv tiktoken")
    exit(1)

load_dotenv()

# --- Configuration ---
# Define project structure paths relative to the script location.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_EMBED_DIR = ROOT_DIR / "embeddings"

# --- OpenAI Client Initialization ---
# Initialize OpenAI client if the API key is available.
try:
    client = OpenAI() # Reads OPENAI_API_KEY from environment variables automatically
    OPENAI_ENABLED = True
except Exception:
    client = None
    OPENAI_ENABLED = False
    print("⚠️ Warning: OpenAI API key not found or invalid. OpenAI models will be skipped.")

# --- Embedding Functions ---
def get_tokenizer(model_id: str) -> tiktoken.Encoding:
    """Get the tokenizer for a given OpenAI model, with a fallback."""
    try:
        return tiktoken.encoding_for_model(model_id)
    except KeyError:
        print(f"⚠️ Warning: Tokenizer for '{model_id}' not found. Using 'cl100k_base'.")
        return tiktoken.get_encoding("cl100k_base")

def generate_embeddings(texts: list[str], model_id: str, batch_size: int, embed_type: str) -> np.ndarray:
   
    if embed_type == 'sbert':
        model = SentenceTransformer(model_id)
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=False
        )
    
    if embed_type == 'openai':
        if not OPENAI_ENABLED:
            raise RuntimeError("OpenAI API key is not configured. Cannot generate embeddings.")
        
        MAX_TOKENS = 8191
        tokenizer = get_tokenizer(model_id)

        def truncate(text: str) -> str:
            """Truncate text to fit within OpenAI's token limit."""
            tokens = tokenizer.encode(text)
            return tokenizer.decode(tokens[:MAX_TOKENS]) if len(tokens) > MAX_TOKENS else text

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"OpenAI ({model_id})"):
            batch = [truncate(t) if t else " " for t in texts[i:i + batch_size]]
            response = client.embeddings.create(model=model_id, input=batch)
            all_embeddings.extend([r.embedding for r in response.data])
        return np.array(all_embeddings, dtype=np.float32)

    raise ValueError(f"Unsupported embedding type: {embed_type}")

# --- Model Definitions ---
MODELS = {
    "openai": {
        "model_id": "text-embedding-3-small",
        "batch_size": 500,
        "embed_type": "openai",
        "enabled": OPENAI_ENABLED,
        "description": "OpenAI text-embedding-3-small (1536 dims)"
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-Embedding-4B",
        "batch_size": 128,
        "embed_type": "sbert",
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

# --- Main Execution ---
def main():
    """Main function to orchestrate the embedding generation process."""
    parser = argparse.ArgumentParser(
        description="Generate text embeddings using various models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_DIR / "final_training_dataset.jsonl",
        help="Path to the input dataset (.jsonl file)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of embeddings, overwriting existing files."
    )
    parser.add_argument(
        "--separator",
        type=str,
        default="\n\n---\n\n",
        help="Separator to join context and user intent."
    )
    parser.add_argument(
        "--mode",
        choices=["normal", "user_intent_only", "context_only"],
        default="normal",
        help="Which column(s) to embed: both combined, user_intent alone, or context alone."
    )
    parser.add_argument(
        "--embed-dir",
        type=Path,
        default=DEFAULT_EMBED_DIR,
        help="Directory to write generated embeddings to."
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=f"Comma-separated subset of models to run ({', '.join(MODELS.keys())}). Default: all."
    )
    args = parser.parse_args()
    EMBED_DIR = args.embed_dir
    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    if args.models:
        selected = {m.strip() for m in args.models.split(",")}
        unknown = selected - set(MODELS.keys())
        if unknown:
            print(f"❌ Error: Unknown model(s) in --models: {', '.join(unknown)}. Available: {', '.join(MODELS.keys())}")
            return
        for key in MODELS:
            if key not in selected:
                MODELS[key]["enabled"] = False

    # --- Data Loading and Validation ---
    if not args.data.is_file():
        print(f"❌ Error: Data file not found at: {args.data}")
        return

    print(f"🔄 Loading data from: {args.data.name}...")
    try:
        df = pd.read_json(args.data, lines=True)
    except ValueError as e:
        print(f"❌ Error reading JSONL file: {e}")
        return

    print("📊 Dataset Statistics:")
    print(f"   - Total samples: {len(df):,}")
    if 'label' in df.columns:
        malicious_count = df['label'].sum()
        total_count = len(df)
        print(f"   - Malicious: {malicious_count:,} ({malicious_count/total_count:.1%})")
        print(f"   - Benign: {total_count - malicious_count:,} ({(total_count - malicious_count)/total_count:.1%})")

    # --- Text Preparation ---
    if args.mode == "user_intent_only":
        print("\n🔗 Mode: user_intent_only — embedding user_intent alone")
        combined_texts = df["user_intent"].astype(str).tolist()
    elif args.mode == "context_only":
        print("\n🔗 Mode: context_only — embedding context alone")
        combined_texts = df["context"].astype(str).tolist()
    else:
        print(f"\n🔗 Mode: normal — combining context and user intent using separator: '{args.separator.strip()}'")
        combined_texts = [
            f"{context}{args.separator}{intent}"
            for context, intent in zip(df["context"].astype(str), df["user_intent"].astype(str))
        ]

    print("📝 Sample combined text (first 200 chars):")
    print(f"   '{combined_texts[0][:200]}...'")

    # --- Embedding Generation Loop ---
    print("\n🚀 Starting embedding generation...")
    for key, config in MODELS.items():
        if not config["enabled"]:
            print(f"\n🟡 Skipping '{key}' model (disabled).")
            continue

        print(f"\n--- ⏳ Processing: {key.upper()} ---")
        print(f"    Model: {config['description']}")
        
        output_path = EMBED_DIR / f"{key}_prompt.npy"

        if output_path.exists() and not args.force:
            print(f"  ↪️  Embeddings already exist. Use --force to overwrite.")
            continue

        try:
            print("  - Generating embeddings...")
            embeddings = generate_embeddings(
                texts=combined_texts,
                model_id=config["model_id"],
                batch_size=config["batch_size"],
                embed_type=config["embed_type"]
            )
            
            np.save(output_path, embeddings)
            print(f"  ✔️  Saved embeddings to: {output_path.name}")
            print(f"      Shape: {embeddings.shape}")
            print(f"      Size: {embeddings.nbytes / 1e6:.2f} MB")
            
        except Exception as e:
            print(f"  ❌ Error generating embeddings for '{key}': {e}")

    print("\n\n🎉🎉🎉 Embedding generation completed! 🎉🎉🎉")
    
    # --- Summary of Generated Files ---
    print("\n📁 Summary of embedding files:")
    for key, config in MODELS.items():
        if config["enabled"]:
            output_path = EMBED_DIR / f"{key}_prompt.npy"
            if output_path.exists():
                size_mb = output_path.stat().st_size / 1e6
                print(f"   ✔️  {output_path.name:<25} ({size_mb:.2f} MB)")
            else:
                print(f"   ❌ {output_path.name:<25} (generation failed or skipped)")

if __name__ == "__main__":
    main()
