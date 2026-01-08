# AGENTS.md

Guidelines for AI coding agents working in this repository.

## Project Overview

Multilingual translation model (Korean/English/Japanese) using Gemma3-270m with LoRA fine-tuning and Bias-corrected EMA. Training completed on 7.4M examples.

## Build & Run Commands

### Environment Setup
```bash
uv sync                          # Install all dependencies
```

### Dataset Preparation
```bash
# Prepare full dataset
uv run python scripts/prepare_simple_translation.py \
  --data_dir /path/to/multilingual-translate \
  --output ./data/processed/simple_translation_full

# Prepare small test dataset
uv run python scripts/prepare_simple_translation.py \
  --data_dir /path/to/multilingual-translate \
  --output ./data/processed/test_dataset \
  --max_samples_per_pair 100
```

### Training
```bash
# Train with EMA (recommended)
uv run python scripts/train_with_ema.py \
  --base_model google/gemma-3-270m \
  --dataset ./data/processed/simple_translation_full \
  --output ./models/adapters/my-translator \
  --epochs 1

# Resume from checkpoint
uv run python scripts/train_with_ema.py \
  --resume_from_checkpoint ./models/adapters/translator-full-ema/checkpoint-XXXXX
```

### Testing
```bash
# Run test suite
uv run python scripts/test_simple_translation.py --mode test

# Interactive mode
uv run python scripts/test_simple_translation.py --mode interactive

# Test specific adapter
uv run python scripts/test_simple_translation.py \
  --adapter ./models/adapters/translator-full-ema \
  --mode test
```

### Run Single Script
```bash
# Run any script directly
uv run python scripts/<script_name>.py

# Run with specific arguments
uv run python scripts/test_simple_translation.py --mode test --max_new_tokens 128
```

## Code Style Guidelines

### File Structure
```
scripts/
├── prepare_simple_translation.py  # Dataset preparation
├── train_simple_translation.py    # Basic training
├── train_with_ema.py              # Training with EMA (primary)
├── test_simple_translation.py     # Model testing
├── ema_utils.py                   # EMA implementation
├── lora_config.py                 # LoRA/quantization config
└── deprecated/                    # Old autotext scripts
```

### Imports
Order: standard library → third-party → local modules. Separate with blank lines.

```python
import os
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from lora_config import get_lora_config, get_bnb_config
```

### Type Hints
Use type hints for function signatures. Optional for local variables.

```python
def translate(
    model,
    tokenizer,
    text: str,
    src_lang: str,
    tgt_lang: str,
    max_new_tokens: int = 256
) -> str:
```

### Docstrings
Use Google-style docstrings for public functions.

```python
def get_lora_config(r: int = 16, lora_alpha: int = 32) -> LoraConfig:
    """
    Get LoRA configuration for translation training.

    Args:
        r: LoRA rank (default: 16)
        lora_alpha: LoRA scaling factor (default: 32)

    Returns:
        LoraConfig instance
    """
```

### Naming Conventions
- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Error Handling
Use explicit error messages. Fail fast for invalid inputs.

```python
if not all_train_examples:
    raise ValueError("No training data loaded! Check data_dir path.")
```

### Print Statements
Use checkmarks and clear formatting for progress output.

```python
print(f"✓ Model loaded")
print(f"  Model dtype: {base_model.dtype}")
print(f"  Device map: {base_model.hf_device_map}")
```

### PyTorch Best Practices
- Use `torch.no_grad()` for inference
- Use `bfloat16` for Gemma models
- Use `device_map="auto"` for automatic GPU placement

```python
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128)
```

## Translation Format

Input/output format used throughout the codebase:

```
<src:SOURCE_LANG><tgt:TARGET_LANG>
SOURCE_TEXT
###
TARGET_TEXT
```

Language codes: `ko`, `en`, `ja`

## Key Constants

| Parameter | Default | Description |
|-----------|---------|-------------|
| LoRA Rank | 16 | Rank of LoRA matrices |
| LoRA Alpha | 32 | Scaling factor |
| LoRA Dropout | 0.05 | Dropout rate |
| EMA Decay | 0.999 | EMA smoothing factor |
| Max Seq Length | 512 | Maximum token length |
| Batch Size | 4 | Per-device batch size |
| Learning Rate | 5e-5 | AdamW learning rate |

## Dependencies

- Python 3.11+
- PyTorch 2.9+ with CUDA 12.8
- transformers 4.55.4
- peft >= 0.17.1
- trl >= 0.22.2

## Important Notes

1. **Base Model Only**: Always use `google/gemma-3-270m` (Base), not instruct variants
2. **EMA Weights**: Final model automatically uses bias-corrected EMA weights
3. **Checkpointing**: Training can be interrupted and resumed from checkpoints
4. **GPU Memory**: ~7GB VRAM with 4-bit quantization
5. **Deprecated**: `scripts/deprecated/` contains old autotext-based code (do not use)

## Testing Changes

Before committing:
1. Run test suite: `uv run python scripts/test_simple_translation.py --mode test`
2. Verify model loads correctly
3. Check translation quality on sample inputs
