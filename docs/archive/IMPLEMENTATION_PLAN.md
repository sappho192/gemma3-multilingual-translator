# IMPLEMENTATION PLAN: Gemma3 Multilingual Translator

## Project Vision & Architecture

This project aims to build a **production-ready multilingual translation system** that:

1. **Handles Multiple Language Pairs**: en↔ko, ja↔ko, ko↔ja (6 translation directions)
2. **Preserves Game UI Elements**: Using autotext token system (`<<AT:id>>`)
3. **Balances General + Domain-Specific Performance**: Through two-stage LoRA training and weighted merging
4. **Enables Web Deployment**: Via ONNX conversion with quantization

## Key Technical Innovations

### 1. Autotext Token Protection System

- **Problem**: Game UI elements like button text, menu items need to be preserved exactly during translation
- **Solution**: Pre-bake special ASCII tokens (`<<AT:`, `>>`) into the base model
- **Why ASCII**: Unicode special tokens can cause encoding issues; ASCII is safer
- **Training Strategy**: Model learns to "copy" these patterns by seeing identical autotext in both source and target during training

### 2. Two-Stage LoRA Training Strategy

```
Base Model (google/gemma-3-270m)
    ↓
LoRA A (General Translation)
    - Mixed language pairs: ko↔en, ja↔en, ko↔ja
    - Broad coverage of translation patterns
    - Establishes basic multilingual capability
    ↓
LoRA B (Game-Specific)
    - FFXIV-specific terminology and dialogue
    - 10-20% replay samples from general data (prevents catastrophic forgetting)
    - Refines translation for gaming context
```

**Why not train together?** Separation allows:
- Independent tuning of each domain
- Easier A/B testing
- Flexible weighted merging ratios

### 3. Weighted Adapter Merging (TIES/DARE)

Instead of simple merging, uses sophisticated algorithms:

**TIES (TrIm, Elect, and Sign merge)**:
- Trims low-magnitude parameters
- Elects top-k parameters when conflicts occur
- Resolves sign conflicts through voting

**DARE (Drop And REscale)**:
- Randomly drops parameters with probability (1 - density)
- Rescales remaining parameters
- Creates sparser, more efficient merged adapters

**Configuration**:
```python
weights = [1.0, 1.2]  # Slightly favor game-specific (B)
density = 0.2         # Keep 20% of parameters
combination_type = "ties"
```

This allows hyperparameter search across weight ratios to find optimal balance.

### 4. Decoding Guard State Machine

Prevents LLM from generating malformed autotext:

```
State 0 (Normal): Any token allowed
    ↓ sees "<<AT:"
State 1 (After Opening): ONLY digits allowed
    ↓ sees digit(s)
State 2 (Reading ID): digits OR ">>" allowed
    ↓ sees ">>"
Back to State 0
```

**Critical**: Without this guard, models might generate:
- `<<AT:>>` (missing ID)
- `<<AT:abc>>` (non-numeric ID)
- `<<AT:123` (unclosed tag)

## Data Processing Pipeline

### Input → Preprocessing
```
Raw: "<autotext>おはようございます！</autotext>久しぶり〜"
         ↓ normalize_autotext with (lang, phrase) → id lookup
Normalized: "<<AT:1023>>久しぶり〜"
```

### Model Format
```
<src:ja><tgt:ko>
<<AT:1023>>久しぶり〜
###
<<AT:1023>>오랜만이야~
```

### Output → Postprocessing
```
Model output: "<<AT:1023>>오랜만이야~"
         ↓ restore_autotext with (ui_lang, id) → phrase lookup
Final (for UI lang=en): "Good morning!오랜만이야~"
```

## Dataset Strategy

### Available Resources

- `~/dataset/aihub_merged/en_ko`: ~163MB train, 23MB val (sourceString, targetString)
- `~/dataset/aihub_merged/ja_ko`: ~100MB train, 13MB val
- `~/dataset/aihub_merged/ko_en`: Similar size
- `~/dataset/aihub_merged/ko_ja`: Similar size

### Training Data Composition

**Stage 1 (General LoRA A)**:
- 40% ko↔en (bidirectional)
- 40% ja↔ko (bidirectional)
- 20% ko↔ja (direct, no pivot)
- Include synthetic autotext examples (10-15% of samples)
- Total: ~100K-200K samples

**Stage 2 (Game-Specific LoRA B)**:
- 70% FFXIV game data (when available)
- 20% replay from Stage 1 general data (catastrophic forgetting prevention)
- 10% synthetic game-specific autotext patterns
- Total: ~20K-50K samples

## Critical Design Decisions

### Why Base Model, Not Instruct?

- **Instruct models** have system/user/assistant formatting baked in
- **Base models** are more flexible for custom task formatting
- We need full control over `<src:><tgt:>` prefix format

### Why No modules_to_save in LoRA?

- Traditional LoRA fine-tuning saves embedding layer changes separately
- We pre-bake tokens into the base model instead
- Cleaner adapter management, easier merging

### Why Separate Pre/Post Processing?

- **Decoupling**: Model only sees normalized `<<AT:id>>` format
- **Flexibility**: Same model can restore different UI languages
- **Safety**: If restoration fails, we can fall back to original source text

## Implementation Roadmap

### Phase 1: Foundation

1. **Create project directory structure** (scripts/, data/, models/, notebooks/)
2. **Add required dependencies** to pyproject.toml:
   - transformers>=4.43
   - peft>=0.12
   - datasets
   - accelerate
   - sentencepiece
   - torch
   - onnx_ir
   - onnxruntime
   - matplotlib
3. **Create scripts/bake_special_tokens.py** - Add autotext tokens (`<<AT:`, `>>`) to base Gemma3-270m model
4. **Create scripts/autotext_utils.py** - Implement normalize_autotext and restore_autotext functions
5. **Create scripts/format_dataset.py** - Format translation data with `<src:lang><tgt:lang>` tags and `###` separator
6. **Create scripts/prepare_datasets.py** - Load and merge aihub_merged datasets for general translation training
7. **Create scripts/lora_config.py** - Configure LoRA settings (r=16, alpha=32, dropout=0.05, target_modules='all-linear', no modules_to_save)

### Phase 2: Training Infrastructure

8. **Create scripts/train_general_lora.py** - Train general translator LoRA (A) on ko↔en, ja↔en, ko↔ja mixed data
9. **Create scripts/train_game_lora.py** - Train game-specific LoRA (B) with replay samples
10. **Create scripts/merge_adapters.py** - Implement weighted adapter merging using TIES/DARE with add_weighted_adapter

### Phase 3: Inference & Safety

11. **Create scripts/decoding_guard.py** - Implement prefix_allowed_tokens_fn state machine for autotext pattern protection
12. **Create scripts/inference.py** - Complete inference pipeline with pre/post-processing
13. **Create scripts/evaluate.py** - Evaluation metrics (autotext preservation, COMET, chrF)

### Phase 4: Deployment & Documentation

14. **Adapt build_gemma.py** from gemma3-exercise for ONNX conversion
15. **Create notebooks/01_bake_tokens.ipynb** - Interactive notebook for token baking
16. **Create notebooks/02_train_general.ipynb** - Training notebook for general LoRA
17. **Create notebooks/03_train_game.ipynb** - Training notebook for game-specific LoRA
18. **Create notebooks/04_merge_and_test.ipynb** - Merging and testing notebook
19. **Update main.py** with CLI interface for common operations
20. **Update README.md** with project overview, setup instructions, and usage examples
21. **Verify CLAUDE.md** completeness and add any missing implementation details

## Evaluation Strategy

### Metrics to Track

1. **Autotext Preservation Rate**: `len(re.findall(r'<<AT:\d+>>', output)) / expected_count`
2. **Pattern Integrity**: All autotext patterns pass regex validation
3. **Translation Quality**:
   - COMET score (reference-based)
   - chrF score (character-level F-score)
4. **Terminology Consistency**: Game-specific glossary match rate
5. **Format Preservation**: Numbers, brackets, punctuation alignment

### A/B Testing Grid for Weighted Merging

```python
weights_grid = [
    (1.0, 1.0),   # Equal weight
    (1.0, 1.2),   # Favor game (recommended)
    (1.0, 1.5),   # Strong game bias
    (0.8, 1.2),   # Reduce general influence
]
```

Test each configuration on held-out dev sets for both general and game domains.

## Risk Mitigation

### Risk 1: Catastrophic Forgetting
When training game-specific LoRA, the model might forget general translation capability.

**Mitigation**: Include 10-20% replay samples from general data in game-specific training.

### Risk 2: Autotext Pattern Corruption
Model might generate malformed autotext patterns during inference.

**Mitigation**:
- Decoding guard (prefix_allowed_tokens_fn)
- Post-generation regex validation
- Fallback to source text if pattern is invalid

### Risk 3: Overfitting to Training Formats
Model might only work with exact training data format.

**Mitigation**: Data augmentation with:
- Spacing variations around autotext
- Emoji insertions near autotext
- Different punctuation patterns

### Risk 4: Wrong Language Output
Model outputs wrong target language despite correct prefix.

**Mitigation**:
- Strong `<src:><tgt:>` prefix training with many examples
- Consider adding language detection post-check
- Validation on prefix-language consistency

## Expected Performance

### General Translation
- COMET: 0.75-0.85 (good quality)
- chrF: 50-60 (character-level alignment)

### Game-Specific
- COMET: 0.70-0.80 (slightly lower due to domain shift)
- Terminology accuracy: 85%+
- Autotext preservation: 98%+

### Inference Speed
ONNX q4f16 on CPU:
- ~10-20 tokens/sec
- Suitable for real-time game translation

## Project Structure

```
gemma3-multilingual-translator/
├── scripts/
│   ├── bake_special_tokens.py       # Token baking into base model
│   ├── autotext_utils.py            # Pre/post-processing utilities
│   ├── format_dataset.py            # Dataset formatting
│   ├── prepare_datasets.py          # Dataset loading and merging
│   ├── lora_config.py               # LoRA configuration
│   ├── train_general_lora.py        # General translator training
│   ├── train_game_lora.py           # Game-specific training
│   ├── merge_adapters.py            # Weighted adapter merging
│   ├── decoding_guard.py            # Constrained decoding
│   ├── inference.py                 # Inference pipeline
│   └── evaluate.py                  # Evaluation metrics
├── notebooks/
│   ├── 01_bake_tokens.ipynb         # Token baking notebook
│   ├── 02_train_general.ipynb       # General training notebook
│   ├── 03_train_game.ipynb          # Game training notebook
│   └── 04_merge_and_test.ipynb      # Merging and testing
├── data/
│   ├── autotext_mappings/           # Autotext ID mappings
│   └── processed/                   # Processed datasets
├── models/
│   ├── base_with_at_tokens/         # Base model with baked tokens
│   ├── adapters/
│   │   ├── translator-general/      # General LoRA adapters
│   │   └── translator-game/         # Game-specific LoRA adapters
│   └── merged_translator_at/        # Final merged model
├── build_gemma.py                   # ONNX conversion script
├── main.py                          # CLI interface
├── pyproject.toml                   # Project dependencies
├── CLAUDE.md                        # Claude Code guidance
├── IMPLEMENTATION_GUIDE.md          # Original guide (Korean)
├── IMPLEMENTATION_PLAN.md           # This file
└── README.md                        # Project documentation
```

## Quick Reference Commands

### Setup
```bash
# Install dependencies
uv sync

# Set up Hugging Face authentication
export HF_TOKEN="your_token_here"
```

### Token Baking (One-time)
```bash
python scripts/bake_special_tokens.py
```

### Training
```bash
# Stage 1: General translation
python scripts/train_general_lora.py \
  --base_model ./models/base_with_at_tokens \
  --output ./models/adapters/translator-general \
  --data_dir ~/dataset/aihub_merged \
  --epochs 3

# Stage 2: Game-specific
python scripts/train_game_lora.py \
  --base_model ./models/base_with_at_tokens \
  --output ./models/adapters/translator-game \
  --game_data ./data/ffxiv \
  --replay_ratio 0.2 \
  --epochs 3
```

### Merging
```bash
python scripts/merge_adapters.py \
  --base ./models/base_with_at_tokens \
  --adapter_a ./models/adapters/translator-general \
  --adapter_b ./models/adapters/translator-game \
  --weights 1.0 1.2 \
  --combination_type ties \
  --density 0.2 \
  --output ./models/merged_translator_at
```

### ONNX Conversion
```bash
uv run build_gemma.py \
  -m ./models/merged_translator_at \
  -o ./models/merged_translator_at_onnx \
  -p fp32 fp16 q4 q4f16
```

### Inference
```bash
python scripts/inference.py \
  --model ./models/merged_translator_at \
  --src_lang ja \
  --tgt_lang ko \
  --text "<<AT:1023>>久しぶり〜" \
  --ui_lang en
```

## Next Steps

1. **Start with Phase 1** to establish the foundation
2. **Create synthetic autotext data** for initial testing
3. **Train on general data first** to validate the pipeline
4. **Iterate on hyperparameters** using small data subsets
5. **Scale up** once the pipeline is validated
6. **Deploy ONNX models** for production use

---

This is a **production-grade architecture** that balances flexibility, safety, and performance. Each component has a specific purpose in the overall system, and the design decisions are optimized for the unique requirements of game translation with UI element preservation.
