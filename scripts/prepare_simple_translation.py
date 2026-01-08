"""
Prepare multilingual translation datasets WITHOUT autotext features.

This script loads translation pairs from JSONL files and formats them
for standard translation training (ko↔en, ja↔ko, en↔ja).

Copyright 2025
Licensed under the Apache License, Version 2.0
"""

import argparse
import json
import random
from pathlib import Path
from datasets import Dataset, DatasetDict
from tqdm import tqdm


def format_translation_example(source: str, target: str, src_lang: str, tgt_lang: str) -> str:
    """
    Format a single translation example.

    Format:
    <src:lang><tgt:lang>
    SOURCE_TEXT
    ###
    TARGET_TEXT

    Args:
        source: Source text
        target: Target text
        src_lang: Source language code (ja, ko, en)
        tgt_lang: Target language code (ja, ko, en)

    Returns:
        Formatted text string
    """
    prefix = f"<src:{src_lang}><tgt:{tgt_lang}>"
    formatted = f"{prefix}\n{source.strip()}\n###\n{target.strip()}"
    return formatted


def load_jsonl_dataset(file_path: Path, src_lang: str, tgt_lang: str, max_samples: int = None):
    """
    Load translation pairs from JSONL file.

    Args:
        file_path: Path to JSONL file
        src_lang: Source language code
        tgt_lang: Target language code
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of formatted examples
    """
    examples = []

    print(f"  Loading {file_path.name}...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break

            data = json.loads(line)
            source = data['sourceString']
            target = data['targetString']

            # Format the example
            formatted = format_translation_example(source, target, src_lang, tgt_lang)
            examples.append({"text": formatted})

    print(f"    Loaded {len(examples)} examples")
    return examples


def prepare_multilingual_dataset(
    data_dir: str,
    output_dir: str,
    max_samples_per_pair: int = None,
    val_split_ratio: float = 0.1,
    seed: int = 42
):
    """
    Prepare multilingual translation dataset from JSONL files.

    Loads all 6 language pairs:
    - en_ja, en_ko
    - ja_en, ja_ko
    - ko_en, ko_ja

    Args:
        data_dir: Base directory containing language pair folders
        output_dir: Output directory for processed dataset
        max_samples_per_pair: Maximum samples per language pair (None for all)
        val_split_ratio: Validation split ratio (default: 0.1)
        seed: Random seed for reproducibility
    """
    random.seed(seed)

    print("\n" + "="*60)
    print("PREPARING MULTILINGUAL TRANSLATION DATASET (NO AUTOTEXT)")
    print("="*60)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Max samples per pair: {max_samples_per_pair or 'ALL'}")
    print(f"Validation split: {val_split_ratio*100}%")

    data_path = Path(data_dir)

    # Define language pairs
    language_pairs = [
        ("en_ja", "en", "ja"),
        ("en_ko", "en", "ko"),
        ("ja_en", "ja", "en"),
        ("ja_ko", "ja", "ko"),
        ("ko_en", "ko", "en"),
        ("ko_ja", "ko", "ja"),
    ]

    all_train_examples = []
    all_val_examples = []

    print("\n=== Loading Training Data ===")
    for pair_dir, src_lang, tgt_lang in language_pairs:
        pair_path = data_path / pair_dir

        if not pair_path.exists():
            print(f"  Warning: {pair_dir} not found, skipping...")
            continue

        print(f"\n{pair_dir} ({src_lang} → {tgt_lang}):")

        # Load train
        train_file = pair_path / "train.jsonl"
        if train_file.exists():
            train_examples = load_jsonl_dataset(
                train_file,
                src_lang,
                tgt_lang,
                max_samples_per_pair
            )
            all_train_examples.extend(train_examples)

        # Load validation
        val_file = pair_path / "val.jsonl"
        if val_file.exists():
            # Use smaller subset for validation
            val_max = int(max_samples_per_pair * val_split_ratio) if max_samples_per_pair else None
            val_examples = load_jsonl_dataset(
                val_file,
                src_lang,
                tgt_lang,
                val_max
            )
            all_val_examples.extend(val_examples)

    if not all_train_examples:
        raise ValueError("No training data loaded! Check data_dir path.")

    # Shuffle
    print("\n=== Shuffling Data ===")
    random.shuffle(all_train_examples)
    random.shuffle(all_val_examples)

    print(f"  Total train examples: {len(all_train_examples):,}")
    print(f"  Total validation examples: {len(all_val_examples):,}")

    # Create datasets
    print("\n=== Creating Dataset Objects ===")
    train_dataset = Dataset.from_list(all_train_examples)
    val_dataset = Dataset.from_list(all_val_examples)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    # Save to disk
    print("\n=== Saving Dataset ===")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_dict.save_to_disk(str(output_path))
    print(f"✓ Dataset saved to {output_path}")

    # Print statistics
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE!")
    print("="*60)
    print(f"\nTrain examples: {len(train_dataset):,}")
    print(f"Validation examples: {len(val_dataset):,}")
    print(f"Total examples: {len(train_dataset) + len(val_dataset):,}")

    # Show samples
    print("\n=== Sample Examples ===")
    for i in range(min(3, len(train_dataset))):
        print(f"\n--- Example {i+1} ---")
        print(train_dataset[i]['text'])

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(
        description="Prepare simple multilingual translation dataset (no autotext)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/tikim/dataset/multilingual-translate",
        help="Base directory containing language pair folders with JSONL files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed/simple_translation",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--max_samples_per_pair",
        type=int,
        default=None,
        help="Maximum samples per language pair (None for all)"
    )
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    prepare_multilingual_dataset(
        data_dir=args.data_dir,
        output_dir=args.output,
        max_samples_per_pair=args.max_samples_per_pair,
        val_split_ratio=args.val_split_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
