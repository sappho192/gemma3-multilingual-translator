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
```

### Stage 1: Token Baking and General Translation Training

**1. Bake special tokens into base model:**
```bash
uv run python scripts/bake_special_tokens.py
```
Output: `./models/base_with_at_tokens/` (262,146 vocab, +2 tokens)

**2. Prepare training dataset:**
```bash
uv run python scripts/prepare_datasets.py
```
Creates: `./data/processed/general_translation/` with 6,210 train / 690 val examples

**3. Train general translation LoRA:**
```bash
uv run python scripts/train_general_lora.py \
  --base_model ./models/base_with_at_tokens \
  --dataset ./data/processed/general_translation \
  --output ./models/adapters/translator-general \
  --epochs 3 \
  --batch_size 2 \
  --learning_rate 5e-5
```
Output: `./models/adapters/translator-general/` (Stage 1 complete)

**4. Test the model:**
```bash
# Run test suite
uv run python scripts/test_model.py --mode test

# Interactive testing
uv run python scripts/test_model.py --mode interactive

# Both test + interactive
uv run python scripts/test_model.py --mode both
```

### Stage 2: Game-Specific Training (Pending)

**1. Train game-specific LoRA:**
```bash
# (Not yet implemented)
uv run python scripts/train_game_lora.py \
  --base_model ./models/base_with_at_tokens \
  --dataset ./data/processed/game_translation \
  --output ./models/adapters/translator-game \
  --replay_samples 0.2
```

**2. Merge adapters with weighted combination:**
```bash
# (Not yet implemented)
uv run python scripts/merge_adapters.py \
  --base ./models/base_with_at_tokens \
  --adapter_a ./models/adapters/translator-general \
  --adapter_b ./models/adapters/translator-game \
  --weights 1.0 1.2 \
  --combination_type ties \
  --output ./merged_translator_at
```

### ONNX Model Building (Future)
```bash
# (build_gemma.py not yet implemented)
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

**Implemented Scripts:**

- `scripts/bake_special_tokens.py` - Add autotext tokens to base model ✓
- `scripts/autotext_utils.py` - Normalize/restore autotext patterns ✓
- `scripts/format_dataset.py` - Format training data with language tags ✓
- `scripts/lora_config.py` - LoRA configuration setup ✓
- `scripts/prepare_datasets.py` - Dataset preparation and loading ✓
- `scripts/train_general_lora.py` - General translation LoRA training (Stage 1) ✓
- `scripts/test_model.py` - Model testing suite with interactive mode ✓

**Pending Scripts:**

- `scripts/merge_adapters.py` - Weighted adapter merging (TIES/DARE) for Stage 2
- `scripts/decoding_guard.py` - Constrained decoding for autotext preservation (high priority)
- `scripts/train_game_lora.py` - Game-specific LoRA training (Stage 2)

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

Current dependencies (from pyproject.toml):
- `transformers==4.55.4`
- `peft>=0.17.1`
- `datasets>=4.2.0`
- `accelerate>=1.10.1`
- `sentencepiece>=0.2.1`
- `torch>=2.9.0` (with CUDA 12.8 support)
- `bitsandbytes>=0.48.1` (for 4-bit quantization)
- `trl>=0.22.2` (for SFTTrainer)
- `evaluate>=0.4.6`
- `tensorboard>=2.20.0`
- `onnx>=1.17.0`, `onnxruntime>=1.23.1` (for future ONNX export)

## Important Constraints

1. **Always use Base model** (google/gemma-3-270m), not instruct variants
2. **Bake special tokens first** before any LoRA training
3. **No modules_to_save** in LoRA configs (tokens already in base)
4. **Preserve autotext in both input and output** during training (model learns to "copy")
5. **Apply decoding guard** during inference to prevent pattern corruption
6. **Include replay samples** when training game-specific LoRA to prevent catastrophic forgetting

## Current Project Status

**Stage 1: General Translation ✓ COMPLETE**

- Token baking: ✓ Complete (262,146 vocab)
- Dataset preparation: ✓ Complete (6,900 examples, ko↔en, ja↔ko, ko↔ja)
- General LoRA training: ✓ Complete (3 epochs, checkpoint-2331)
- Model testing: ✓ Complete (89% translation accuracy, 6 language pairs)

**Test Results Summary:**
- 16/18 test cases passed with good translation quality
- Excellent performance on JA↔EN (100%), KO→JA (100%), KO→EN (100%)
- Issue identified: Autotext token preservation needs decoding guard implementation

**Stage 2: Game-Specific Training (Pending)**

Next steps:
1. Implement `scripts/decoding_guard.py` (HIGH PRIORITY - autotext preservation)
2. Collect/prepare FFXIV game-specific translation data
3. Implement `scripts/train_game_lora.py` with replay samples
4. Implement `scripts/merge_adapters.py` for weighted LoRA merging
5. Final evaluation and ONNX export

## Evaluation Metrics

**Implemented:**
- Manual test suite with 18 test cases across 6 language pairs
- Translation quality assessment (excellent/good/fair/poor)
- Language pair-specific success rates
- Interactive testing mode for ad-hoc validation

**Pending (for Stage 2):**
- Autotext preservation rate (regex pattern matching) - requires decoding guard
- Number/bracket/tag preservation accuracy
- Terminology consistency (glossary matching)
- COMET / chrF scores for quantitative evaluation
