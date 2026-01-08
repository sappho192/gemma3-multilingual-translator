# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **multilingual translation system** based on Gemma3-270m trained on 7.4M translation pairs across Korean, English, and Japanese. The model uses LoRA fine-tuning with Bias-corrected EMA for improved generalization.

- **Base Model**: google/gemma-3-270m (Base, not instruct)
- **Training**: LoRA with Bias-corrected EMA on 7.4M examples
- **Languages**: Korean (ko), English (en), Japanese (ja) - all 6 directions
- **Final Loss**: 1.59 (from initial 4.03)
- **Inference**: Supports both PyTorch and ONNX (PyTorch-free)

## Quick Start

```bash
# Install dependencies
uv sync

# Test the trained model (PyTorch)
uv run python scripts/test_simple_translation.py \
  --adapter ./models/adapters/translator-full-ema \
  --mode interactive

# Or test with ONNX (no PyTorch needed after conversion)
uv run python scripts/test_onnx_translation.py \
  --model_dir ./models/onnx/translator \
  --precision q4 \
  --mode interactive
```

## Development Commands

### Dataset Preparation

```bash
# Prepare multilingual dataset from JSONL files
uv run python scripts/prepare_simple_translation.py \
  --data_dir /path/to/multilingual-translate \
  --output ./data/processed/simple_translation_full \
  --max_samples_per_pair 1500000
```

### Training

```bash
# Train with EMA (recommended)
uv run python scripts/train_with_ema.py \
  --base_model google/gemma-3-270m \
  --dataset ./data/processed/simple_translation_full \
  --output ./models/adapters/translator-full-ema \
  --epochs 1 \
  --batch_size 4 \
  --gradient_accumulation_steps 4

# Resume training from checkpoint
uv run python scripts/train_with_ema.py \
  --base_model google/gemma-3-270m \
  --dataset ./data/processed/simple_translation_full \
  --output ./models/adapters/translator-full-ema \
  --resume_from_checkpoint ./models/adapters/translator-full-ema/checkpoint-XXXXX
```

### Testing (PyTorch)

```bash
# Run test suite
uv run python scripts/test_simple_translation.py \
  --adapter ./models/adapters/translator-full-ema \
  --mode test

# Interactive testing
uv run python scripts/test_simple_translation.py \
  --adapter ./models/adapters/translator-full-ema \
  --mode interactive
```

### ONNX Conversion

```bash
# Convert LoRA adapter to ONNX (merges weights)
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

# Interactive mode
uv run python scripts/test_onnx_translation.py \
  --model_dir ./models/onnx/translator \
  --mode interactive

# Benchmark mode
uv run python scripts/test_onnx_translation.py \
  --model_dir ./models/onnx/translator \
  --mode benchmark
```

## Project Structure

```
gemma3-multilingual-translator/
├── scripts/
│   ├── prepare_simple_translation.py  # Dataset preparation
│   ├── train_simple_translation.py    # Basic training (no EMA)
│   ├── train_with_ema.py              # Training with EMA (recommended)
│   ├── test_simple_translation.py     # PyTorch model testing
│   ├── convert_to_onnx.py             # ONNX conversion
│   ├── test_onnx_translation.py       # ONNX model testing
│   ├── ema_utils.py                   # EMA implementation
│   ├── lora_config.py                 # LoRA/quantization config
│   └── deprecated/                    # Old autotext-based scripts
├── torch_free/                        # PyTorch-free ONNX inference
│   ├── __init__.py
│   ├── requirements.txt               # Minimal dependencies
│   ├── inference/
│   │   ├── translator_inference.py    # Main TranslatorInferencer class
│   │   ├── gemma_session.py           # ONNX Runtime session manager
│   │   ├── gemma_tokenizer.py         # Tokenizer wrapper
│   │   ├── kv_cache.py                # KV cache management
│   │   └── generation.py              # Greedy/sampling algorithms
│   └── examples/
│       └── simple_translate.py        # Usage example
├── data/
│   └── processed/                     # Prepared datasets
├── models/
│   ├── adapters/
│   │   └── translator-full-ema/       # LoRA adapter
│   └── onnx/
│       └── translator/                # ONNX models
├── docs/                              # Documentation archive
└── resume_training.sh                 # Resume script
```

## Training Configuration

### Current Model (translator-full-ema)

| Parameter | Value |
|-----------|-------|
| Base Model | google/gemma-3-270m |
| Training Samples | 7,404,947 |
| Validation Samples | 906,530 |
| Epochs | 1 |
| Batch Size | 4 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 16 |
| Learning Rate | 5e-5 |
| Max Sequence Length | 512 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| EMA Decay | 0.999 |
| Total Steps | 462,500 |

### Training Results

- **Initial Loss**: 4.03
- **Final Loss**: 1.59
- **Loss Reduction**: 60%
- **Training Time**: ~70 hours

### ONNX Model Specifications

| Precision | Model Size | Avg. Latency | Notes |
|-----------|------------|--------------|-------|
| fp32 | 1.1 GB | 0.26s | Best accuracy |
| q4 | 764 MB | 0.17s | Recommended |

## Translation Format

Input format for translation:
```
<src:SOURCE_LANG><tgt:TARGET_LANG>
SOURCE_TEXT
###
```

Example:
```
<src:ko><tgt:en>
안녕하세요, 만나서 반갑습니다.
###
```

Output will be the translated text after `###`.

## Model Usage

### PyTorch API

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load model
base = AutoModelForCausalLM.from_pretrained(
    'google/gemma-3-270m',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained('./models/adapters/translator-full-ema')
model = PeftModel.from_pretrained(base, './models/adapters/translator-full-ema')
model.eval()

# Translate
prompt = "<src:ko><tgt:en>\n안녕하세요\n###\n"
inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
translation = result.split('###')[-1].strip()
print(translation)
```

### ONNX API (No PyTorch)

```python
from torch_free.inference import TranslatorInferencer

# Load ONNX model
translator = TranslatorInferencer(
    "./models/onnx/translator",
    precision="q4"  # or fp32, fp16, q4f16
)

# Translate
result = translator.translate(
    "안녕하세요, 만나서 반갑습니다.",
    src_lang="ko",
    tgt_lang="en"
)
print(result)  # "Hello, nice to meet you."

# Batch translation
results = translator.batch_translate(
    ["안녕하세요", "감사합니다"],
    src_lang="ko",
    tgt_lang="en"
)
```

## Dependencies

### Full (Training + PyTorch Inference)

From `pyproject.toml`:
- `transformers>=4.55.4`
- `peft>=0.17.1`
- `datasets>=4.2.0`
- `accelerate>=1.10.1`
- `torch>=2.9.0` (CUDA 12.8)
- `bitsandbytes>=0.48.1`
- `trl>=0.22.2`
- `onnx-ir>=0.1.11` (for conversion)

### Minimal (ONNX Inference Only)

From `torch_free/requirements.txt`:
- `numpy>=1.24.0`
- `onnxruntime>=1.16.0`
- `tokenizers>=0.15.0`

## Deprecated Features

The following features from the original design are deprecated and moved to `scripts/deprecated/`:

- **Autotext token system** (`<<AT:id>>` markers for game UI phrases)
- **Token baking** into base model
- **Decoding guard** for autotext preservation
- **Two-stage training** (general + game-specific)
- **Weighted adapter merging** (TIES/DARE)

These may be revisited for game-specific translation in the future.

## Notes

- Always use the Base model (not instruct variants)
- EMA weights are automatically applied to the final saved model
- Training can be interrupted and resumed from checkpoints
- GPU memory usage: ~7GB VRAM with 4-bit quantization
- ONNX conversion requires PyTorch to merge LoRA weights
- ONNX inference runs without PyTorch using ONNX Runtime
