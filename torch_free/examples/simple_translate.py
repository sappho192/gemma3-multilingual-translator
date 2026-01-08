#!/usr/bin/env python3
"""
Simple translation example using PyTorch-free ONNX inference.

Usage:
    python torch_free/examples/simple_translate.py \
        --model_dir ./models/onnx/translator \
        --text "안녕하세요" \
        --src ko \
        --tgt en
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from torch_free.inference import TranslatorInferencer


def main():
    parser = argparse.ArgumentParser(description="Simple translation example")
    parser.add_argument("--model_dir", required=True, help="Path to ONNX model directory")
    parser.add_argument("--text", required=True, help="Text to translate")
    parser.add_argument("--src", required=True, choices=["ko", "en", "ja"], help="Source language")
    parser.add_argument("--tgt", required=True, choices=["ko", "en", "ja"], help="Target language")
    parser.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "q4", "q4f16"])

    args = parser.parse_args()

    # Initialize inferencer
    print(f"Loading model from {args.model_dir}...")
    inferencer = TranslatorInferencer(
        model_dir=args.model_dir,
        precision=args.precision,
        verbose=False
    )

    # Translate
    print(f"\nTranslating: {args.text}")
    print(f"Direction: {args.src} -> {args.tgt}")

    result = inferencer.translate(
        text=args.text,
        src_lang=args.src,
        tgt_lang=args.tgt,
        max_new_tokens=128
    )

    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
