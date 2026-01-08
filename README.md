# Gemma3 Multilingual Translator

A multilingual translation model based on Gemma3-270m, fine-tuned with LoRA and Bias-corrected EMA on 7.4M translation pairs.

## Features

- **6-way translation**: Korean, English, Japanese (all directions)
- **Lightweight**: LoRA adapter (~15MB) on top of Gemma3-270m base
- **High quality**: 60% loss reduction (4.03 → 1.59) with EMA smoothing
- **Easy to use**: Simple prompt format with language tags

## Quick Start

```bash
# Install dependencies
uv sync

# Interactive translation
uv run python scripts/test_simple_translation.py \
  --adapter ./models/adapters/translator-full-ema \
  --mode interactive
```

## Usage

### Python API

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

# Translate Korean to English
prompt = "<src:ko><tgt:en>\n안녕하세요, 만나서 반갑습니다.\n###\n"
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
print(translation)  # "Hello, nice to meet you."
```

### Translation Format

```
<src:SOURCE_LANG><tgt:TARGET_LANG>
SOURCE_TEXT
###
```

Language codes: `ko` (Korean), `en` (English), `ja` (Japanese)

### Examples

| Direction | Input | Output |
|-----------|-------|--------|
| KO→EN | 안녕하세요, 만나서 반갑습니다. | Hello, nice to meet you. |
| EN→KO | Hello, nice to meet you. | 안녕하세요, 만나서 반갑습니다. |
| JA→KO | こんにちは、お元気ですか？ | 안녕하세요, 건강하세요? |
| KO→JA | 오늘 날씨가 정말 좋네요. | 今日の天気は本当にいいですね。 |

## Training

### Model Specifications

| Parameter | Value |
|-----------|-------|
| Base Model | google/gemma-3-270m |
| Training Data | 7.4M translation pairs |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| EMA Decay | 0.999 |
| Final Loss | 1.59 |

### Train Your Own

```bash
# Prepare dataset
uv run python scripts/prepare_simple_translation.py \
  --data_dir /path/to/translation-data \
  --output ./data/processed/my_dataset

# Train with EMA
uv run python scripts/train_with_ema.py \
  --base_model google/gemma-3-270m \
  --dataset ./data/processed/my_dataset \
  --output ./models/adapters/my-translator \
  --epochs 1
```

## Project Structure

```
gemma3-multilingual-translator/
├── scripts/
│   ├── prepare_simple_translation.py  # Dataset preparation
│   ├── train_with_ema.py              # Training with EMA
│   ├── test_simple_translation.py     # Model testing
│   ├── ema_utils.py                   # EMA implementation
│   └── lora_config.py                 # LoRA configuration
├── models/adapters/
│   └── translator-full-ema/           # Trained model
└── data/processed/                    # Prepared datasets
```

## Requirements

- Python 3.11+
- PyTorch 2.9+ with CUDA
- ~7GB GPU VRAM (with 4-bit quantization)

## License

Apache License 2.0
