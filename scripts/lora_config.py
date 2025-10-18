"""
LoRA configuration for translation model training.

Defines the LoRA configuration optimized for Gemma3-270m translation tasks.

Copyright 2025
Licensed under the Apache License, Version 2.0
"""

from peft import LoraConfig
from transformers import BitsAndBytesConfig
import torch


def get_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    use_modules_to_save: bool = False
) -> LoraConfig:
    """
    Get LoRA configuration for translation training.

    IMPORTANT: For this project, we do NOT use modules_to_save because
    special tokens are pre-baked into the base model.

    Args:
        r: LoRA rank (default: 16)
        lora_alpha: LoRA scaling factor (default: 32)
        lora_dropout: Dropout rate (default: 0.05)
        use_modules_to_save: Whether to save embedding layers (default: False)

    Returns:
        LoraConfig instance
    """
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules="all-linear",
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Only add modules_to_save if explicitly requested
    # (We don't use this for our autotext approach)
    if use_modules_to_save:
        config.modules_to_save = ["lm_head", "embed_tokens"]

    return config


def get_bnb_config(
    load_in_4bit: bool = True,
    compute_dtype: torch.dtype = torch.bfloat16
) -> BitsAndBytesConfig:
    """
    Get BitsAndBytes quantization configuration for memory-efficient training.

    Args:
        load_in_4bit: Whether to load model in 4-bit (default: True)
        compute_dtype: Computation dtype (default: bfloat16)

    Returns:
        BitsAndBytesConfig instance
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype
    )


def print_lora_config(config: LoraConfig):
    """Print LoRA configuration in a readable format."""
    print("\n=== LoRA Configuration ===")
    print(f"  Rank (r): {config.r}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Dropout: {config.lora_dropout}")
    print(f"  Target modules: {config.target_modules}")
    print(f"  Bias: {config.bias}")
    print(f"  Task type: {config.task_type}")
    print(f"  Modules to save: {config.modules_to_save if hasattr(config, 'modules_to_save') else 'None'}")
    print()


if __name__ == "__main__":
    # Test configuration
    print("Testing LoRA configuration...")

    lora_config = get_lora_config()
    print_lora_config(lora_config)

    bnb_config = get_bnb_config()
    print("=== BitsAndBytes Configuration ===")
    print(f"  Load in 4-bit: {bnb_config.load_in_4bit}")
    print(f"  Quantization type: {bnb_config.bnb_4bit_quant_type}")
    print(f"  Compute dtype: {bnb_config.bnb_4bit_compute_dtype}")
