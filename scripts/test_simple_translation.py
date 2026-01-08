"""
Test simple translation model WITHOUT autotext features.

This script tests the trained LoRA adapter for translation quality
across multiple language pairs.

Copyright 2025
Licensed under the Apache License, Version 2.0
"""

import os
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import login
from dotenv import load_dotenv


def setup_hf_auth():
    """Set up Hugging Face authentication"""
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if hf_token:
        login(hf_token)
        print("✓ Logged in to Hugging Face")


def load_model(base_model_name: str, adapter_path: str = None):
    """
    Load model with optional LoRA adapter.

    Args:
        base_model_name: Base model name or path
        adapter_path: Path to LoRA adapter (None for base model only)

    Returns:
        (model, tokenizer)
    """
    print(f"\nLoading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)

    print(f"Loading base model: {base_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation='eager'
    )

    if adapter_path:
        print(f"Loading LoRA adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        print("✓ Model with LoRA adapter loaded")
    else:
        print("✓ Base model loaded (no adapter)")

    # Set pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.eval()
    return model, tokenizer


def translate(
    model,
    tokenizer,
    text: str,
    src_lang: str,
    tgt_lang: str,
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    top_p: float = 0.9
) -> str:
    """
    Translate text from source to target language.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
        text: Source text to translate
        src_lang: Source language code (ja, ko, en)
        tgt_lang: Target language code (ja, ko, en)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        Translated text
    """
    # Format prompt
    prompt = f"<src:{src_lang}><tgt:{tgt_lang}>\n{text.strip()}\n###\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract translation (after ###)
    if "###" in generated_text:
        translation = generated_text.split("###", 1)[1].strip()
    else:
        translation = generated_text.strip()

    return translation


def run_test_suite(model, tokenizer):
    """
    Run comprehensive test suite.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
    """
    print("\n" + "="*60)
    print("TRANSLATION TEST SUITE")
    print("="*60)

    test_cases = [
        # Korean to English
        {
            "text": "안녕하세요, 만나서 반갑습니다.",
            "src": "ko",
            "tgt": "en",
            "reference": "Hello, nice to meet you."
        },
        {
            "text": "오늘 날씨가 정말 좋네요.",
            "src": "ko",
            "tgt": "en",
            "reference": "The weather is really nice today."
        },
        {
            "text": "이 책은 매우 흥미로웠습니다.",
            "src": "ko",
            "tgt": "en",
            "reference": "This book was very interesting."
        },

        # English to Korean
        {
            "text": "How are you today?",
            "src": "en",
            "tgt": "ko",
            "reference": "오늘 어떻게 지내세요?"
        },
        {
            "text": "I love learning new languages.",
            "src": "en",
            "tgt": "ko",
            "reference": "저는 새로운 언어를 배우는 것을 좋아합니다."
        },

        # Japanese to Korean
        {
            "text": "今日はいい天気ですね。",
            "src": "ja",
            "tgt": "ko",
            "reference": "오늘은 날씨가 좋네요."
        },
        {
            "text": "ありがとうございます。",
            "src": "ja",
            "tgt": "ko",
            "reference": "감사합니다."
        },

        # Korean to Japanese
        {
            "text": "감사합니다. 잘 부탁드립니다.",
            "src": "ko",
            "tgt": "ja",
            "reference": "ありがとうございます。よろしくお願いします。"
        },

        # English to Japanese
        {
            "text": "Good morning! How are you?",
            "src": "en",
            "tgt": "ja",
            "reference": "おはようございます！お元気ですか？"
        },

        # Japanese to English
        {
            "text": "私は日本語を勉強しています。",
            "src": "ja",
            "tgt": "en",
            "reference": "I am studying Japanese."
        },
    ]

    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test {i}/{len(test_cases)} ---")
        print(f"Direction: {test['src']} → {test['tgt']}")
        print(f"Source: {test['text']}")
        print(f"Reference: {test['reference']}")

        translation = translate(
            model, tokenizer, test['text'], test['src'], test['tgt']
        )

        print(f"Translation: {translation}")

        # Simple quality assessment (can be improved)
        quality = "✓" if len(translation) > 0 else "✗"
        results.append(quality)
        print(f"Status: {quality}")

    # Summary
    passed = results.count("✓")
    total = len(results)

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Failed: {total-passed}/{total}")


def interactive_mode(model, tokenizer):
    """
    Interactive translation mode.

    Args:
        model: Model instance
        tokenizer: Tokenizer instance
    """
    print("\n" + "="*60)
    print("INTERACTIVE TRANSLATION MODE")
    print("="*60)
    print("\nLanguage codes: ko (Korean), en (English), ja (Japanese)")
    print("Example: 'ko en Hello, how are you?'")
    print("Type 'quit' to exit\n")

    while True:
        try:
            user_input = input("Input (src tgt text): ").strip()

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print("Error: Please provide src_lang, tgt_lang, and text")
                continue

            src_lang, tgt_lang, text = parts

            if src_lang not in ['ko', 'en', 'ja'] or tgt_lang not in ['ko', 'en', 'ja']:
                print("Error: Language codes must be ko, en, or ja")
                continue

            translation = translate(model, tokenizer, text, src_lang, tgt_lang)
            print(f"\nTranslation: {translation}\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Test simple translation model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-3-270m",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./models/adapters/translator-full-ema",
        help="Path to LoRA adapter (omit to test base model only)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "interactive", "both"],
        help="Test mode: test suite, interactive, or both"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature"
    )

    args = parser.parse_args()

    # Setup
    setup_hf_auth()

    # Load model
    model, tokenizer = load_model(args.base_model, args.adapter)

    # Run tests
    if args.mode in ["test", "both"]:
        run_test_suite(model, tokenizer)

    if args.mode in ["interactive", "both"]:
        interactive_mode(model, tokenizer)


if __name__ == "__main__":
    main()
