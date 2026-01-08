"""
Merge LoRA adapter with base model and save as SafeTensors.

This script:
1. Loads the base Gemma3-270m model
2. Merges the LoRA adapter weights into the base model
3. Saves the merged model in SafeTensors format

Usage:
    uv run python scripts/merge_lora_to_safetensor.py \
        --adapter ./models/adapters/translator-full-ema \
        --output ./models/merged/translator-full

Copyright 2025. Licensed under Apache License 2.0.
"""

import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel


def merge_and_save(
    adapter_path: str,
    output_dir: str,
    base_model: str = "google/gemma-3-270m",
    dtype: str = "bfloat16"
):
    """
    Merge LoRA adapter with base model and save as SafeTensors.

    Args:
        adapter_path: Path to LoRA adapter directory
        output_dir: Output directory for merged model
        base_model: Base model name (default: google/gemma-3-270m)
        dtype: Model dtype (default: bfloat16)
    """
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    print(f"Loading base model: {base_model}")
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map="cpu",
    )
    print(f"  Model dtype: {base_model_obj.dtype}")
    print(f"  Parameters: {base_model_obj.num_parameters():,}")

    print(f"\nLoading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model_obj, adapter_path)

    print("\nMerging LoRA adapter with base model...")
    merged_model = model.merge_and_unload()
    print(f"  Merged model dtype: {merged_model.dtype}")
    print(f"  Merged parameters: {merged_model.num_parameters():,}")

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving merged model to: {output_dir}")
    merged_model.save_pretrained(output_dir, safe_serialization=True)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_dir)

    print("Saving config...")
    config = AutoConfig.from_pretrained(base_model)
    config.save_pretrained(output_dir)

    gen_config_path = os.path.join(adapter_path, "generation_config.json")
    if os.path.exists(gen_config_path):
        print("Saving generation config from adapter...")
        generation_config = GenerationConfig.from_pretrained(adapter_path)
    else:
        print("Creating default generation config...")
        generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generation_config.save_pretrained(output_dir)

    print(f"\n{'='*60}")
    print("Merge complete!")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print("\nOutput files:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            print(f"  {f}: {size / (1024 * 1024):.1f} MB")
        elif size > 1024:
            print(f"  {f}: {size / 1024:.1f} KB")
        else:
            print(f"  {f}: {size} B")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapter with base model and save as SafeTensors",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--adapter",
        type=str,
        default="./models/adapters/translator-full-ema",
        help="Path to LoRA adapter directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/merged/translator-full",
        help="Output directory for merged model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-3-270m",
        help="Base model name (default: google/gemma-3-270m)"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Model dtype (default: bfloat16)"
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Gemma3 Multilingual Translator - LoRA Merger")
    print(f"{'='*60}")
    print(f"Adapter: {args.adapter}")
    print(f"Output: {args.output}")
    print(f"Base Model: {args.base_model}")
    print(f"Dtype: {args.dtype}")
    print(f"{'='*60}\n")

    merge_and_save(
        adapter_path=args.adapter,
        output_dir=args.output,
        base_model=args.base_model,
        dtype=args.dtype
    )


if __name__ == "__main__":
    main()
