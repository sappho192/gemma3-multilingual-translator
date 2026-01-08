# AGENTS.md

Guidelines for AI coding agents working in this repository.

## Project Overview

Multilingual translation model (Korean/English/Japanese) using Gemma3-270m with LoRA fine-tuning and Bias-corrected EMA. Training completed on 7.4M examples. Supports both PyTorch and ONNX inference.

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

### Testing (PyTorch)
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

### ONNX Conversion
```bash
# Convert LoRA adapter to ONNX
uv run python scripts/convert_to_onnx.py \
  --adapter ./models/adapters/translator-full-ema \
  --output ./models/onnx/translator \
  --precision fp32 q4

# Available precisions: fp32, fp16, q4, q4f16
```

### Testing (ONNX - No PyTorch)
```bash
# Run ONNX test suite
uv run python scripts/test_onnx_translation.py \
  --model_dir ./models/onnx/translator \
  --precision q4 \
  --mode test

# ONNX interactive mode
uv run python scripts/test_onnx_translation.py \
  --model_dir ./models/onnx/translator \
  --mode interactive

# ONNX benchmark
uv run python scripts/test_onnx_translation.py \
  --model_dir ./models/onnx/translator \
  --mode benchmark
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
├── test_simple_translation.py     # PyTorch model testing
├── convert_to_onnx.py             # ONNX conversion
├── test_onnx_translation.py       # ONNX model testing
├── ema_utils.py                   # EMA implementation
├── lora_config.py                 # LoRA/quantization config
└── deprecated/                    # Old autotext scripts

torch_free/
├── __init__.py
├── requirements.txt               # Minimal deps for ONNX inference
├── inference/
│   ├── __init__.py
│   ├── translator_inference.py    # Main TranslatorInferencer class
│   ├── gemma_session.py           # ONNX Runtime session manager
│   ├── gemma_tokenizer.py         # Tokenizer wrapper
│   ├── kv_cache.py                # KV cache management
│   └── generation.py              # Greedy/sampling algorithms
└── examples/
    └── simple_translate.py        # Usage example
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

### ONNX Best Practices
- Use numpy arrays with explicit dtypes (int64, float32)
- Initialize KV cache with zeros for first inference step
- Use q4 precision for best speed/size tradeoff

```python
from torch_free.inference import TranslatorInferencer

translator = TranslatorInferencer("./models/onnx/translator", precision="q4")
result = translator.translate("Hello", src_lang="en", tgt_lang="ko")
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

## ONNX Model Specifications

| Precision | Model Size | Avg. Latency | Notes |
|-----------|------------|--------------|-------|
| fp32 | 1.1 GB | 0.26s | Best accuracy |
| fp16 | ~600 MB | ~0.20s | Good balance |
| q4 | 764 MB | 0.17s | Recommended for deployment |
| q4f16 | ~400 MB | ~0.15s | Smallest, fastest |

## Dependencies

### Full (Training + PyTorch)
- Python 3.11+
- PyTorch 2.9+ with CUDA 12.8
- transformers 4.55.4
- peft >= 0.17.1
- trl >= 0.22.2
- onnx-ir >= 0.1.11 (for conversion)

### Minimal (ONNX Inference)
- Python 3.11+
- numpy >= 1.24.0
- onnxruntime >= 1.16.0
- tokenizers >= 0.15.0

## Important Notes

1. **Base Model Only**: Always use `google/gemma-3-270m` (Base), not instruct variants
2. **EMA Weights**: Final model automatically uses bias-corrected EMA weights
3. **Checkpointing**: Training can be interrupted and resumed from checkpoints
4. **GPU Memory**: ~7GB VRAM with 4-bit quantization
5. **Deprecated**: `scripts/deprecated/` contains old autotext-based code (do not use)
6. **ONNX Conversion**: Requires PyTorch to merge LoRA weights before export
7. **ONNX Inference**: No PyTorch needed after conversion - uses ONNX Runtime only

## Testing Changes

Before committing:

### PyTorch Model
```bash
uv run python scripts/test_simple_translation.py --mode test
```

### ONNX Model
```bash
uv run python scripts/test_onnx_translation.py \
  --model_dir ./models/onnx/translator \
  --precision q4 \
  --mode test
```

Both should report "All PASSED" for the translation tests.
