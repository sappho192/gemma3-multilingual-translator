# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **multilingual translation system** based on Gemma3-270m that targets game-specific translation (specifically FFXIV) while maintaining general translation capabilities. The architecture uses:

- **Base Model**: google/gemma-3-270m (Base, not instruct)
- **Training Strategy**: Two-stage LoRA training (general → game-specific) with weighted merging
- **Special Feature**: Autotext token protection system for preserving game UI phrases (using `<<AT:id>>` markers)

The project implements a sophisticated pipeline:
1. Special token "baking" into base model
2. Two separate LoRA adapters (general translation + game-specific)
3. Weighted adapter merging using TIES/DARE algorithms
4. Decoding guard to prevent autotext pattern corruption
5. Pre/post-processing for autotext normalization and restoration

## Development Commands

### Environment Setup
```bash
# Python 3.11+ required (see .python-version)
uv sync  # Install dependencies with uv
# OR
pip install -r requirements.txt  # If requirements.txt exists
```

### Running the Application
```bash
python main.py  # Entry point (currently minimal)
# OR
uv run main.py
```

### ONNX Model Building
```bash
uv run build_gemma.py \
  -m ./merged_translator_at \
  -o ./merged_translator_at_onnx \
  -p fp32 fp16 q4 q4f16
```

## Architecture Details

### Special Token System
The model uses ASCII-based autotext markers (`<<AT:` and `>>`) that are:
- Added to the **base model only** (not LoRA adapters)
- Used to protect game-specific UI phrases during translation
- Preserved through a custom decoding guard (`prefix_allowed_tokens_fn`)

**Critical**: modules_to_save is NOT used for LoRA training since tokens are pre-baked into base.

### Two-Stage LoRA Training

**LoRA Configuration**:
- r=16, alpha=32, dropout=0.05
- target_modules="all-linear"
- No modules_to_save

**Stage 1 - General Translator (LoRA A)**:
- Training data: Mixed ko↔en, ja↔en, ko↔ja with autotext examples
- Output: `adapters/translator-general`

**Stage 2 - Game-Specific (LoRA B)**:
- Training data: FFXIV chat/dialogue/UI + 10-20% replay samples from general
- Output: `adapters/translator-game`

### Weighted Adapter Merging
Uses PEFT's `add_weighted_adapter` with:
- **combination_type**: "ties" or "dare"
- **weights**: [1.0, 1.2] (slightly favor game-specific)
- **density**: 0.2
- Final merged model saved to `./merged_translator_at`

### Training Input Format
```
<src:{src_lang}><tgt:{tgt_lang}>
{source_text}
###
{target_text}
```

Example:
```
<src:ja><tgt:ko>
<<AT:1023>>久しぶり〜
###
<<AT:1023>>오랜만이야~
```

### Decoding Guard Implementation
Custom `prefix_allowed_tokens_fn` ensures autotext patterns remain valid:
- State machine tracks: 0=normal, 1=after `<<AT:`, 2=reading digits
- Forces digit tokens after opening, only allows `>>` or more digits during number reading
- Prevents model from generating malformed patterns like `<<AT:>>` or `<<AT:abc>>`

## Key Scripts Structure

Based on IMPLEMENTATION_GUIDE.md, the expected scripts are:

- `scripts/bake_special_tokens.py` - Add autotext tokens to base model
- `scripts/autotext_utils.py` - Normalize/restore autotext patterns
- `scripts/format_dataset.py` - Format training data with language tags
- `scripts/lora_config.py` - LoRA configuration setup
- `scripts/merge_adapters.py` - Weighted adapter merging (TIES/DARE)
- `scripts/decoding_guard.py` - Constrained decoding for autotext preservation

## Data Processing Pipeline

### Preprocessing (normalize_autotext)
```
<autotext>おはようございます！</autotext>久しぶり〜
↓
<<AT:1023>>久しぶり〜
```

### Model Inference
Input → Model with decoding guard → Output with preserved `<<AT:id>>`

### Postprocessing (restore_autotext)
```
<<AT:1023>>오랜만이야~
↓ (for UI language: en)
Good morning!오랜만이야~
```

## Related Resources

- **Base repository**: `~/repo/gemma3-exercise` (contains original training notebooks/scripts)
- **Translation datasets**: `~/aihub-translation-dataset`, `~/dataset/aihub_merged`
- See IMPLEMENTATION_GUIDE.md for detailed step-by-step implementation instructions (in Korean)

## Model Dependencies

Required packages (add to pyproject.toml dependencies):
- `transformers>=4.43`
- `peft>=0.12`
- `datasets`
- `accelerate`
- `sentencepiece`
- `torch`

## Important Constraints

1. **Always use Base model** (google/gemma-3-270m), not instruct variants
2. **Bake special tokens first** before any LoRA training
3. **No modules_to_save** in LoRA configs (tokens already in base)
4. **Preserve autotext in both input and output** during training (model learns to "copy")
5. **Apply decoding guard** during inference to prevent pattern corruption
6. **Include replay samples** when training game-specific LoRA to prevent catastrophic forgetting

## Evaluation Metrics

When testing weighted merge configurations:
- Autotext preservation rate (regex pattern matching)
- Number/bracket/tag preservation accuracy
- Terminology consistency (glossary matching)
- COMET / chrF scores
