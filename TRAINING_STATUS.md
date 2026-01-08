# Training Status: COMPLETED

## Final Results

| Metric | Value |
|--------|-------|
| Status | **COMPLETED** |
| Total Steps | 462,500 / 462,810 |
| Initial Loss | 4.03 |
| Final Loss | 1.59 |
| Loss Reduction | 60% |
| Training Time | ~70 hours |

## Model Location

```
./models/adapters/translator-full-ema/
├── adapter_config.json
├── adapter_model.safetensors (14.5MB)
├── tokenizer.json
├── tokenizer.model
├── training_loss.png
└── runs/ (TensorBoard logs)
```

## Training Configuration

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

## Test Results

| Direction | Input | Output | Status |
|-----------|-------|--------|--------|
| KO→EN | 안녕하세요, 만나서 반갑습니다. | Hello, I'm here to meet you. | OK |
| EN→KO | Hello, nice to meet you. | 안녕하세요, 만나서 반갑습니다. | Good |
| JA→KO | こんにちは、お元気ですか？ | 안녕하세요, 건강하세요? | Good |
| KO→JA | 오늘 날씨가 정말 좋네요. | 今日の天気は本当にいいですね。 | Good |
| JA→EN | 今日は天気がいいですね。 | Today is a good weather. | Minor grammar |
| EN→JA | The weather is nice today. | 今日の天気はいいですね。 | Good |

## Usage

```bash
# Test the model
uv run python scripts/test_simple_translation.py \
  --adapter ./models/adapters/translator-full-ema \
  --mode interactive
```
