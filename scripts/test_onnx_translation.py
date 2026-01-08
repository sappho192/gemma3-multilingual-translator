"""
Test ONNX Translation Model without PyTorch dependency.

This script tests the ONNX-converted translation model using only
ONNX Runtime and numpy (no PyTorch required).

Usage:
    uv run python scripts/test_onnx_translation.py \
        --model_dir ./models/onnx/translator \
        --precision q4 \
        --mode test

    # Interactive mode
    uv run python scripts/test_onnx_translation.py \
        --model_dir ./models/onnx/translator \
        --mode interactive
"""

import argparse
import sys
import time
from pathlib import Path

# Add torch_free to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from torch_free.inference import TranslatorInferencer


def test_translation(inferencer: TranslatorInferencer) -> bool:
    """
    Run translation tests.

    Returns:
        True if all tests pass
    """
    test_cases = [
        # Korean to English
        ("안녕하세요, 만나서 반갑습니다.", "ko", "en"),
        ("오늘 날씨가 정말 좋네요.", "ko", "en"),
        ("저는 한국어를 배우고 있습니다.", "ko", "en"),

        # English to Korean
        ("Hello, nice to meet you.", "en", "ko"),
        ("The weather is really nice today.", "en", "ko"),
        ("I am learning Korean.", "en", "ko"),

        # Japanese to Korean
        ("こんにちは、お元気ですか？", "ja", "ko"),
        ("今日の天気は本当にいいですね。", "ja", "ko"),

        # Korean to Japanese
        ("안녕하세요, 건강하세요?", "ko", "ja"),
        ("오늘 날씨가 정말 좋네요.", "ko", "ja"),

        # English to Japanese
        ("Good morning, how are you?", "en", "ja"),
        ("I love Japanese food.", "en", "ja"),

        # Japanese to English
        ("おはようございます。", "ja", "en"),
        ("日本料理が大好きです。", "ja", "en"),
    ]

    print("\n" + "=" * 70)
    print("Running Translation Tests")
    print("=" * 70)

    all_passed = True
    total_time = 0

    for i, (text, src_lang, tgt_lang) in enumerate(test_cases, 1):
        print(f"\n[Test {i}/{len(test_cases)}] {src_lang} -> {tgt_lang}")
        print(f"Input: {text}")

        start_time = time.time()
        try:
            result = inferencer.translate(
                text,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                max_new_tokens=128,
                do_sample=False
            )
            elapsed = time.time() - start_time
            total_time += elapsed

            print(f"Output: {result}")
            print(f"Time: {elapsed:.2f}s")

            # Basic validation: output should not be empty
            if not result or len(result.strip()) == 0:
                print("FAILED: Empty output")
                all_passed = False
            else:
                print("PASSED")

        except Exception as e:
            print(f"FAILED: {e}")
            all_passed = False

    print("\n" + "=" * 70)
    print(f"Test Summary: {'All PASSED' if all_passed else 'Some FAILED'}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per translation: {total_time / len(test_cases):.2f}s")
    print("=" * 70)

    return all_passed


def benchmark(inferencer: TranslatorInferencer, iterations: int = 10):
    """
    Run benchmark tests.
    """
    test_text = "안녕하세요, 오늘 날씨가 정말 좋네요."

    print("\n" + "=" * 70)
    print(f"Running Benchmark ({iterations} iterations)")
    print("=" * 70)
    print(f"Test text: {test_text}")

    # Warmup
    print("\nWarmup...")
    for _ in range(2):
        inferencer.translate(test_text, "ko", "en", max_new_tokens=64)

    # Benchmark
    print("Benchmarking...")
    times = []
    for i in range(iterations):
        start = time.time()
        result = inferencer.translate(test_text, "ko", "en", max_new_tokens=64)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Iteration {i + 1}: {elapsed:.3f}s -> {result[:50]}...")

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)

    print("\n" + "-" * 70)
    print(f"Results:")
    print(f"  Average: {avg_time:.3f}s")
    print(f"  Min: {min_time:.3f}s")
    print(f"  Max: {max_time:.3f}s")
    print("=" * 70)


def interactive_mode(inferencer: TranslatorInferencer):
    """
    Run interactive translation mode.
    """
    inferencer.translate_interactive()


def main():
    parser = argparse.ArgumentParser(
        description="Test ONNX Translation Model",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models/onnx/translator",
        help="Path to ONNX model directory"
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "fp16", "q4", "q4f16"],
        default="fp32",
        help="Model precision to use (default: fp32)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "benchmark", "interactive"],
        default="test",
        help="Test mode: test, benchmark, or interactive"
    )
    parser.add_argument(
        "--benchmark_iterations",
        type=int,
        default=10,
        help="Number of iterations for benchmark mode"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Check model directory exists
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        print("\nTo convert your model to ONNX, run:")
        print("  uv run python scripts/convert_to_onnx.py \\")
        print("    --adapter ./models/adapters/translator-full-ema \\")
        print("    --output ./models/onnx/translator \\")
        print("    --precision fp32 q4")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("Gemma3 Multilingual Translator - ONNX Test")
    print(f"{'=' * 70}")
    print(f"Model directory: {args.model_dir}")
    print(f"Precision: {args.precision}")
    print(f"Mode: {args.mode}")
    print(f"{'=' * 70}\n")

    # Initialize inferencer
    try:
        inferencer = TranslatorInferencer(
            model_dir=str(model_dir),
            precision=args.precision,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Run selected mode
    if args.mode == "test":
        success = test_translation(inferencer)
        sys.exit(0 if success else 1)
    elif args.mode == "benchmark":
        benchmark(inferencer, args.benchmark_iterations)
    elif args.mode == "interactive":
        interactive_mode(inferencer)


if __name__ == "__main__":
    main()
