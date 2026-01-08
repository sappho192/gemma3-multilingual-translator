# Testing Guide: General Translation Model

## Quick Start

The general translation LoRA model has been trained and is ready to test!

### 1. Run Test Suite (Recommended First)

This runs predefined test cases across all language pairs:

```bash
uv run python scripts/test_model.py --mode test
```

**What it tests:**
- English ↔ Korean translations
- Japanese ↔ Korean translations
- English ↔ Japanese translations
- Autotext marker preservation

### 2. Interactive Mode

For manual testing with your own text:

```bash
uv run python scripts/test_model.py --mode interactive
```

Then follow the prompts:
```
Source language (en/ko/ja): en
Target language (en/ko/ja): ko
Enter text in en: Hello, how are you?

Translation: 안녕하세요, 잘 지내세요?
```

### 3. Run Both

```bash
uv run python scripts/test_model.py --mode both
```

## Command Line Options

```bash
uv run python scripts/test_model.py \
  --base_model ./models/base_with_at_tokens \
  --adapter ./models/adapters/translator-general \
  --mode test \
  --max_tokens 128 \
  --temperature 0.0 \
  --num_beams 4
```

### Options:

- `--base_model`: Path to base model with baked autotext tokens (default: `./models/base_with_at_tokens`)
- `--adapter`: Path to trained LoRA adapters (default: `./models/adapters/translator-general`)
- `--mode`: Testing mode
  - `test`: Run predefined test suite
  - `interactive`: Manual testing mode
  - `both`: Run test suite then enter interactive mode
- `--max_tokens`: Maximum tokens to generate (default: 128)
- `--temperature`: Sampling temperature
  - `0.0`: Greedy decoding (deterministic, recommended)
  - `>0.0`: Sampling (more creative but less consistent)
- `--num_beams`: Number of beams for beam search (default: 4, higher = better quality but slower)

## Supported Language Pairs

The model supports 6 translation directions:

1. **English ↔ Korean** (en ↔ ko)
2. **Japanese ↔ Korean** (ja ↔ ko)
3. **English ↔ Japanese** (en ↔ ja)

## Testing Autotext Preservation

The model should preserve autotext markers like `<<AT:1023>>`:

```bash
# Interactive mode example
Source language: en
Target language: ko
Enter text: <<AT:1001>>Hello, welcome to the game!

# Expected: <<AT:1001>> should be preserved in Korean translation
Translation: <<AT:1001>>안녕하세요, 게임에 오신 것을 환영합니다!
```

## Example Test Cases

### Simple Greetings

```
EN→KO: "Hello, how are you?" → "안녕하세요, 어떻게 지내세요?"
JA→KO: "こんにちは、お元気ですか？" → "안녕하세요, 잘 지내세요?"
KO→JA: "안녕하세요, 잘 지내세요?" → "こんにちは、お元気ですか？"
```

### Common Phrases

```
EN→KO: "Thank you very much" → "정말 감사합니다"
JA→EN: "ありがとうございます" → "Thank you very much"
KO→EN: "감사합니다" → "Thank you"
```

### With Context

```
EN→KO: "The weather is nice today." → "오늘 날씨가 좋네요."
JA→KO: "今日はいい天気ですね。" → "오늘은 좋은 날씨네요."
```

## Interpreting Results

### Good Translation Signs:
- ✓ Correct target language
- ✓ Preserves meaning
- ✓ Natural phrasing
- ✓ Autotext markers preserved exactly

### Issues to Watch For:
- ✗ Wrong target language output
- ✗ Incomplete translations
- ✗ Malformed autotext patterns
- ✗ Hallucinations or nonsense

## Performance Tips

### For Better Quality:
- Use `--num_beams 4` or higher (slower but better)
- Use `--temperature 0.0` for consistency
- Keep input texts concise (<512 tokens)

### For Faster Inference:
- Use `--num_beams 1` (greedy only)
- Reduce `--max_tokens` to expected length
- Use smaller batch sizes

## Troubleshooting

### Issue: Model outputs wrong language
**Solution:** The model might need more training data for that direction, or the source language tag might be unclear. Try rephrasing.

### Issue: Autotext patterns get corrupted
**Solution:** This shouldn't happen often. If it does, the decoding guard (future feature) will help.

### Issue: Translations are too literal
**Solution:** This is expected for the general model. The game-specific model (Stage 2) will handle domain-specific phrasing better.

### Issue: Out of memory
**Solution:** Reduce `--num_beams` or restart and try again with fewer parallel processes.

## Training Information

- **Model**: Gemma3-270m base with autotext tokens
- **Training data**: 6,210 examples (90% train / 10% val)
- **Language mix**: 40% ko↔en, 40% ja↔ko, 20% direct pairs
- **Autotext augmentation**: 15% of examples
- **Training time**: ~19 minutes (3 epochs)
- **Final loss**: Check `./models/adapters/translator-general/training_loss.png`

## Next Steps

After testing the general model:

1. **Collect game-specific data** (FFXIV dialogue, UI text)
2. **Train Stage 2** (game-specific LoRA with replay)
3. **Merge adapters** (weighted TIES/DARE merging)
4. **Add decoding guard** (autotext pattern protection)
5. **Convert to ONNX** (web deployment)

## Files & Directories

```
models/
├── base_with_at_tokens/          # Base model + autotext tokens
└── adapters/
    └── translator-general/        # Stage 1 LoRA (you are here)
        ├── adapter_config.json
        ├── adapter_model.safetensors
        ├── training_loss.png
        └── runs/                  # TensorBoard logs
```

## Need Help?

Check:
- `IMPLEMENTATION_PLAN.md` - Overall project architecture
- `STAGE1_SUMMARY.md` - Stage 1 implementation details
- `CLAUDE.md` - Technical guidance for development
