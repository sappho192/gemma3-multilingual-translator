"""
Bake special autotext tokens into the base Gemma3-270m model.

This script adds the autotext special tokens (<<AT:, >>) to the base model's
vocabulary and resizes the token embeddings. This needs to be done ONCE before
any LoRA training.

Copyright 2025
Licensed under the Apache License, Version 2.0
"""

import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from dotenv import load_dotenv


def setup_hf_auth():
    """Set up Hugging Face authentication"""
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not found in .env file")
        print("You may need to authenticate manually if accessing gated models")
    else:
        login(hf_token)
        print("✓ Logged in to Hugging Face")


def bake_tokens(base_model_id: str, output_path: str):
    """
    Add special autotext tokens to the base model.

    Args:
        base_model_id: Hugging Face model ID (e.g., "google/gemma-3-270m")
        output_path: Path to save the model with baked tokens
    """
    print(f"\n=== Baking Special Tokens into {base_model_id} ===")

    # Define special tokens for autotext
    SPECIAL_TOKENS = {
        "additional_special_tokens": ["<<AT:", ">>"]
    }

    print("\n1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    print(f"   Original vocab size: {len(tokenizer)}")

    print("\n2. Adding special tokens...")
    num_added = tokenizer.add_special_tokens(SPECIAL_TOKENS)
    print(f"   Added {num_added} new tokens")
    print(f"   New vocab size: {len(tokenizer)}")
    print(f"   Token IDs: <<AT: = {tokenizer.convert_tokens_to_ids('<<AT:')}, >> = {tokenizer.convert_tokens_to_ids('>>')}")

    print("\n3. Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        device_map="auto",
        torch_dtype="auto"
    )
    print(f"   Original embedding size: {model.get_input_embeddings().weight.shape}")

    print("\n4. Resizing token embeddings...")
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"   New embedding size: {model.get_input_embeddings().weight.shape}")

    print(f"\n5. Saving to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    model.save_pretrained(output_path)

    print("\n✓ Special tokens successfully baked into base model!")
    print(f"✓ Saved to: {output_path}")

    # Verify
    print("\n6. Verification...")
    test_tokenizer = AutoTokenizer.from_pretrained(output_path, use_fast=True)
    test_text = "<<AT:1023>>Hello world>>"
    tokens = test_tokenizer.tokenize(test_text)
    print(f"   Test text: {test_text}")
    print(f"   Tokens: {tokens}")
    print(f"   Token IDs: {test_tokenizer.convert_tokens_to_ids(tokens)}")


def main():
    parser = argparse.ArgumentParser(
        description="Bake autotext special tokens into Gemma3-270m base model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-3-270m",
        help="Base model ID or path (default: google/gemma-3-270m)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/base_with_at_tokens",
        help="Output directory for model with baked tokens"
    )

    args = parser.parse_args()

    # Authenticate with HuggingFace
    setup_hf_auth()

    # Bake the tokens
    bake_tokens(args.base_model, args.output)


if __name__ == "__main__":
    main()
