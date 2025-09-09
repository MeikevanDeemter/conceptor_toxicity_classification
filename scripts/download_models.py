#!/usr/bin/env python3
"""Script to download Hugging Face models."""

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# List of models to download
models = [
    # Small Models (1-3B)
    "gpt2",
    "gpt2-medium",
    "gpt2-large",
    "gpt2-xl",
    "facebook/opt-1.3b",
    "EleutherAI/pythia-1b",
    "EleutherAI/pythia-2.8b",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen3-1.7B",
    # Medium Models (6-13B)
    "EleutherAI/gpt-j-6b",
    "tiiuae/falcon-7b",
    "meta-llama/Llama-2-7b",
    # Large Models (13-30B)
    "EleutherAI/gpt-neox-20b",
    "meta-llama/Llama-2-13b",
]


def download_model(model_id):
    """Download model and tokenizer to default Hugging Face cache."""
    print(f"Downloading {model_id}...")

    # Determine if this is a large model (>6B parameters)
    is_large_model = any(x in model_id.lower() for x in ["neox", "llama-2-13b", "falcon-7b"])

    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print(f"✓ Tokenizer for {model_id} downloaded")
    del tokenizer

    # For large models use half precision and device_map
    if is_large_model:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, device_map="auto", torch_dtype=torch.float16, low_cpu_mem_usage=True
            )
            print(f"✓ Model {model_id} downloaded (with optimizations for large model)")
        except Exception as e:
            print(f"❌ Error downloading {model_id}: {str(e)}")
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id)
            print(f"✓ Model {model_id} downloaded")
            del model
        except Exception as e:
            print(f"❌ Error downloading {model_id}: {str(e)}")

    print(f"Completed download for {model_id}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Hugging Face models for offline use")
    parser.add_argument("--models", nargs="+", help="Specific models to download")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Custom cache directory (uses Hugging Face default if not specified)",
    )
    args = parser.parse_args()

    # Set custom cache directory if provided
    if args.cache_dir:
        os.environ["HF_HUB_CACHE"] = args.cache_dir
        print(f"Using custom cache directory: {args.cache_dir}")

    if args.models:
        models_to_download = args.models
    elif args.all:
        models_to_download = models
    else:
        print("Available models:")
        for i, model in enumerate(models):
            print(f"{i+1}. {model}")

        selection = input(
            "\nEnter model numbers to download (comma-separated, e.g., '1,3,5') or 'all': "
        )

        if selection.lower() == "all":
            models_to_download = models
        else:
            try:
                indices = [int(idx.strip()) - 1 for idx in selection.split(",")]
                models_to_download = [models[idx] for idx in indices if 0 <= idx < len(models)]
            except Exception as e:
                print(f"Invalid selection. Exiting: {e}")
                exit(1)

    print(f"Will download {len(models_to_download)} models: {', '.join(models_to_download)}")
    confirm = input("Continue? (y/n): ")

    if confirm.lower() != "y":
        print("Download canceled.")
        exit(0)

    for model_id in models_to_download:
        download_model(model_id)

    print("All requested models have been downloaded!")
