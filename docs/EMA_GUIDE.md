# Bias-corrected EMA for LoRA Training Guide

## Overview

**Exponential Moving Average (EMA)** with bias correction is now integrated into the translation LoRA training pipeline. EMA maintains a moving average of model parameters during training, which often leads to:

- **Better generalization**: Smoother decision boundaries
- **More stable inference**: Reduced variance in predictions
- **Improved final performance**: Often 0.5-2% better metrics
- **Robustness**: Less sensitive to learning rate spikes

### What is Bias Correction?

Early in training, EMA estimates are biased towards initialization. Bias correction adjusts for this:

```
ema_param = decay * ema_param + (1 - decay) * param
corrected_ema = ema_param / (1 - decay^num_updates)
```

This is similar to the bias correction in Adam optimizer.

## Quick Start

### 1. Basic Training with EMA

```bash
uv run python scripts/train_with_ema.py \
  --base_model google/gemma-3-270m \
  --dataset ./data/processed/simple_translation_10k \
  --output ./models/adapters/translator-with-ema \
  --epochs 3
```

Default EMA settings:
- `decay=0.999` (99.9% weight to old EMA, 0.1% to new param)
- `update_after_step=100` (start EMA after 100 steps)
- EMA weights used for evaluation and final model

### 2. Custom EMA Configuration

```bash
uv run python scripts/train_with_ema.py \
  --base_model google/gemma-3-270m \
  --dataset ./data/processed/simple_translation_10k \
  --output ./models/adapters/translator-ema-custom \
  --ema_decay 0.9999 \
  --ema_min_decay 0.99 \
  --ema_update_after_step 200 \
  --epochs 3
```

**Parameter Guide:**
- `--ema_decay`: Higher = smoother averaging (0.999 - 0.9999)
  - 0.999: Standard, works for most cases
  - 0.9999: Smoother, good for long training
  - 0.995: Faster adaptation, good for short training

- `--ema_min_decay`: Starting decay value for warmup (0.0 - 0.99)
  - 0.0: Start from scratch (default)
  - 0.99: Gentle warmup

- `--ema_update_after_step`: When to start EMA (50 - 500)
  - 100: Default, balanced
  - 200-500: For very noisy early training

### 3. Disable EMA Features

```bash
# Don't use EMA for evaluation (but still maintain it)
uv run python scripts/train_with_ema.py \
  --no_ema_eval \
  ...

# Don't save final model with EMA weights
uv run python scripts/train_with_ema.py \
  --no_save_ema \
  ...
```

## Implementation Details

### How it Works with LoRA

EMA is applied **only to trainable parameters** (LoRA adapters):

1. **During Training**:
   - After each step, update EMA of LoRA parameters
   - Keep original parameters for gradient updates

2. **During Evaluation**:
   - Temporarily replace LoRA parameters with bias-corrected EMA
   - Run evaluation
   - Restore original parameters

3. **Final Model**:
   - Option to save model with EMA parameters applied
   - EMA state saved separately for resuming

### Memory Overhead

EMA doubles the memory for **trainable parameters only**:
- Base model: No extra memory
- LoRA adapters: 2x memory (original + EMA)
- Total overhead: ~0.5-1% for typical LoRA configs

Example for Gemma-3-270m with LoRA r=16:
- Model: ~270M params → ~540MB (bf16)
- LoRA: ~500K params → ~1MB (bf16)
- EMA overhead: ~1MB (LoRA only)

## Advanced Usage

### Loading Model with EMA

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
from ema_utils import load_model_with_ema

# Load base + adapter
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m")
model = PeftModel.from_pretrained(model, "./models/adapters/translator-with-ema")

# Apply saved EMA state
model = load_model_with_ema(
    model,
    "./models/adapters/translator-with-ema/ema_state_final.pt"
)
```

### Manual EMA Integration

```python
from ema_utils import BiasCorrectEMA, EMACallback

# Initialize EMA
ema = BiasCorrectEMA(
    model=trainer.model,
    decay=0.999,
    update_after_step=100
)

# Add callback to trainer
callback = EMACallback(ema, save_ema_weights=True)
trainer.add_callback(callback)

# Train
trainer.train()
```

### Comparing Models with/without EMA

```bash
# Train without EMA
uv run python scripts/train_simple_translation.py \
  --output ./models/adapters/translator-no-ema \
  ...

# Train with EMA
uv run python scripts/train_with_ema.py \
  --output ./models/adapters/translator-with-ema \
  ...

# Test both
uv run python scripts/test_simple_translation.py \
  --adapter ./models/adapters/translator-no-ema \
  --mode test

uv run python scripts/test_simple_translation.py \
  --adapter ./models/adapters/translator-with-ema \
  --mode test
```

## EMA Hyperparameter Tuning

### Recommended Settings by Training Length

**Short training (1-2 epochs, <5000 steps):**
```bash
--ema_decay 0.995 \
--ema_update_after_step 50
```

**Medium training (3-5 epochs, 5000-20000 steps):**
```bash
--ema_decay 0.999 \
--ema_update_after_step 100
```

**Long training (10+ epochs, >20000 steps):**
```bash
--ema_decay 0.9999 \
--ema_update_after_step 200 \
--ema_min_decay 0.99
```

### Effect of Decay Rate

| Decay | Effective Window | Use Case |
|-------|------------------|----------|
| 0.99  | ~100 steps | Very fast adaptation, noisy |
| 0.995 | ~200 steps | Fast adaptation |
| 0.999 | ~1000 steps | **Balanced (recommended)** |
| 0.9999 | ~10000 steps | Very smooth, long training |

The effective window is approximately `1 / (1 - decay)` steps.

## Monitoring EMA

### TensorBoard

EMA statistics are logged automatically:
```bash
tensorboard --logdir ./models/adapters/translator-with-ema/runs
```

Look for:
- Validation loss should be smoother with EMA
- Final eval loss often lower with EMA

### Console Output

During training, EMA callback prints:
```
Saved EMA state to .../ema_state_step_3750.pt
```

At the end:
```
=== Applying EMA weights to final model ===
✓ Final model now uses bias-corrected EMA weights
  Total EMA updates: 11250
  Bias correction factor: 0.999999
```

## Expected Performance Gains

Based on typical translation tasks:

- **BLEU/COMET**: +0.3 to +1.5 points
- **Validation loss**: -0.01 to -0.05 reduction
- **Stability**: Reduced variance across runs
- **Best use case**: Medium to long training runs

**Note**: Short training (<1000 steps) may not benefit significantly from EMA.

## Troubleshooting

### Issue: OOM (Out of Memory)

EMA doubles LoRA parameter memory. Solutions:
1. Reduce LoRA rank: `--lora_r 8` instead of 16
2. Reduce batch size: `--batch_size 2`
3. Increase gradient accumulation: `--gradient_accumulation_steps 8`

### Issue: EMA not improving results

Possible causes:
1. Training too short (< 1000 steps)
2. Decay too high/low for your training length
3. Already at optimal performance

Solutions:
1. Train longer or adjust decay
2. Try decay=0.995 for shorter training
3. Compare carefully with multiple seeds

### Issue: EMA callback errors

Make sure:
1. EMA initialized **after** PEFT model creation
2. Using compatible transformers/PEFT versions
3. Callback added before `trainer.train()`

## Integration with Existing Code

To add EMA to your existing training script:

```python
# 1. Import
from ema_utils import BiasCorrectEMA, EMACallback

# 2. After creating trainer, before train()
ema = BiasCorrectEMA(trainer.model, decay=0.999)
trainer.add_callback(EMACallback(ema))

# 3. Train as normal
trainer.train()

# Done! EMA is now active
```

## Files and Checkpoints

When training with EMA, you'll see:

```
models/adapters/translator-with-ema/
├── adapter_config.json          # LoRA config
├── adapter_model.safetensors    # LoRA weights (with EMA if --save_ema)
├── ema_state_final.pt           # Final EMA state
├── ema_state_step_3750.pt       # EMA checkpoint (epoch 1)
├── ema_state_step_7500.pt       # EMA checkpoint (epoch 2)
├── ema_state_step_11250.pt      # EMA checkpoint (epoch 3)
├── training_loss.png            # Training curves
└── runs/                        # TensorBoard logs
```

## References

- Original EMA paper: [Mean teachers are better role models](https://arxiv.org/abs/1703.01780)
- Bias correction: [Adam optimizer paper](https://arxiv.org/abs/1412.6980)
- LoRA: [Low-Rank Adaptation paper](https://arxiv.org/abs/2106.09685)

## FAQ

**Q: Should I always use EMA?**
A: For most medium/long training runs, yes. For very short experiments (<1000 steps), optional.

**Q: Does EMA slow down training?**
A: Minimal impact (<1% overhead). EMA update is a simple weighted average.

**Q: Can I use EMA with other optimizers?**
A: Yes! EMA is optimizer-agnostic and works with any optimizer.

**Q: What if I want to resume training with EMA?**
A: Load the EMA state file and continue training. EMA maintains its history.

**Q: Does EMA work with full fine-tuning?**
A: Yes, but memory overhead is much higher (2x all parameters, not just LoRA).
