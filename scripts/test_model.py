"""
Test the trained general translation LoRA model.

This script loads the trained LoRA adapter and performs test translations
across multiple language pairs.

Copyright 2025
Licensed under the Apache License, Version 2.0
"""

import os
import argparse
import torch
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


def load_model_with_lora(base_model_path: str, adapter_path: str):
    """
    Load base model and apply LoRA adapters.

    Args:
        base_model_path: Path to base model with baked tokens
        adapter_path: Path to LoRA adapters

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\nLoading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)

    print(f"Loading base model from {base_model_path}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    print(f"Loading LoRA adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("✓ Model and adapters loaded successfully\n")

    return model, tokenizer


def translate(
    model,
    tokenizer,
    source_text: str,
    src_lang: str,
    tgt_lang: str,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    num_beams: int = 4
) -> str:
    """
    Translate text from source language to target language.

    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
        source_text: Text to translate
        src_lang: Source language code (ja, ko, en)
        tgt_lang: Target language code (ja, ko, en)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0.0 for greedy)
        num_beams: Number of beams for beam search

    Returns:
        Translated text
    """
    # Format input with language tags
    prompt = f"<src:{src_lang}><tgt:{tgt_lang}>\n{source_text.strip()}\n###\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else 1.0,
            num_beams=num_beams if temperature == 0.0 else 1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Extract translation (after ###)
    if "###" in generated_text:
        translation = generated_text.split("###")[1].strip()
        # Remove any trailing tokens
        if tokenizer.eos_token in translation:
            translation = translation.split(tokenizer.eos_token)[0].strip()
    else:
        translation = generated_text

    return translation


def run_test_suite(model, tokenizer):
    """
    Run a comprehensive test suite across language pairs.

    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
    """
    print("="*70)
    print("TRANSLATION TEST SUITE")
    print("="*70)

    # Test cases: (source_text, src_lang, tgt_lang, description)
    test_cases = [
        # English → Korean
        ("Hello, how are you?", "en", "ko", "EN→KO: Simple greeting"),
        ("The weather is nice today.", "en", "ko", "EN→KO: Weather"),
        ("I love learning languages.", "en", "ko", "EN→KO: Personal statement"),

        # Korean → English
        ("안녕하세요, 반갑습니다.", "ko", "en", "KO→EN: Greeting"),
        ("오늘 날씨가 정말 좋네요.", "ko", "en", "KO→EN: Weather"),
        ("저는 한국어를 공부하고 있습니다.", "ko", "en", "KO→EN: Learning"),

        # Japanese → Korean
        ("こんにちは、お元気ですか？", "ja", "ko", "JA→KO: Greeting"),
        ("今日はいい天気ですね。", "ja", "ko", "JA→KO: Weather"),
        ("日本語を勉強しています。", "ja", "ko", "JA→KO: Learning"),

        # Korean → Japanese
        ("안녕하세요, 잘 지내세요?", "ko", "ja", "KO→JA: Greeting"),
        ("오늘은 날씨가 좋습니다.", "ko", "ja", "KO→JA: Weather"),
        ("일본어를 배우고 있어요.", "ko", "ja", "KO→JA: Learning"),

        # English → Japanese
        ("Good morning!", "en", "ja", "EN→JA: Morning greeting"),
        ("Thank you very much.", "en", "ja", "EN→JA: Thanks"),

        # Japanese → English
        ("ありがとうございます。", "ja", "en", "JA→EN: Thanks"),
        ("おやすみなさい。", "ja", "en", "JA→EN: Good night"),
    ]

    for i, (source, src_lang, tgt_lang, description) in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {description}")
        print(f"Source ({src_lang}): {source}")

        translation = translate(model, tokenizer, source, src_lang, tgt_lang)

        print(f"Translation ({tgt_lang}): {translation}")
        print("-" * 70)


def interactive_mode(model, tokenizer):
    """
    Run interactive translation mode.

    Args:
        model: Model with LoRA adapters
        tokenizer: Tokenizer
    """
    print("\n" + "="*70)
    print("INTERACTIVE TRANSLATION MODE")
    print("="*70)
    print("\nLanguage codes: en (English), ko (Korean), ja (Japanese)")
    print("Type 'quit' to exit\n")

    while True:
        try:
            # Get source language
            src_lang = input("Source language (en/ko/ja): ").strip().lower()
            if src_lang == 'quit':
                break
            if src_lang not in ['en', 'ko', 'ja']:
                print("Invalid language code. Please use en, ko, or ja.")
                continue

            # Get target language
            tgt_lang = input("Target language (en/ko/ja): ").strip().lower()
            if tgt_lang == 'quit':
                break
            if tgt_lang not in ['en', 'ko', 'ja']:
                print("Invalid language code. Please use en, ko, or ja.")
                continue

            # Get text to translate
            source_text = input(f"Enter text in {src_lang}: ").strip()
            if source_text == 'quit':
                break
            if not source_text:
                print("Please enter some text.")
                continue

            # Translate
            print(f"\nTranslating {src_lang} → {tgt_lang}...")
            translation = translate(model, tokenizer, source_text, src_lang, tgt_lang)

            print(f"Translation: {translation}\n")
            print("-" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Test the trained general translation LoRA model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="./models/base_with_at_tokens",
        help="Path to base model with baked tokens"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="./models/adapters/translator-general",
        help="Path to LoRA adapters"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="test",
        choices=["test", "interactive", "both"],
        help="Mode: 'test' for test suite, 'interactive' for manual testing, 'both' for both"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 for greedy decoding)"
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams for beam search"
    )

    args = parser.parse_args()

    # Setup
    setup_hf_auth()

    # Load model
    model, tokenizer = load_model_with_lora(args.base_model, args.adapter)

    # Run tests
    if args.mode in ["test", "both"]:
        run_test_suite(model, tokenizer)

    if args.mode in ["interactive", "both"]:
        interactive_mode(model, tokenizer)

    print("\n✓ Testing complete!")


if __name__ == "__main__":
    main()
