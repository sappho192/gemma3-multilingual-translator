"""
Prepare and load translation datasets for training.

Loads the aihub_merged datasets and formats them for translation training.

Copyright 2025
Licensed under the Apache License, Version 2.0
"""

import argparse
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from format_dataset import format_translation_batch, augment_with_autotext
import random


def load_aihub_dataset(
    data_dir: str,
    language_pair: str,
    split: str = "train",
    max_samples: int = None
) -> Dataset:
    """
    Load AIHub translation dataset.

    Args:
        data_dir: Base directory containing aihub_merged datasets
        language_pair: Language pair directory (en_ko, ja_ko, ko_en, ko_ja)
        split: 'train' or 'val'
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        Dataset with sourceString and targetString columns
    """
    data_path = Path(data_dir) / language_pair / f"{split}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    print(f"Loading {language_pair} {split} from {data_path}...")

    dataset = load_dataset(
        "csv",
        data_files=str(data_path),
        split="train"  # load_dataset always uses 'train' split name for CSV
    )

    if max_samples and len(dataset) > max_samples:
        print(f"  Sampling {max_samples} from {len(dataset)} examples...")
        indices = random.sample(range(len(dataset)), max_samples)
        dataset = dataset.select(indices)

    print(f"  Loaded {len(dataset)} examples")

    return dataset


def prepare_general_translation_dataset(
    data_dir: str,
    max_samples_per_pair: int = None,
    autotext_augmentation_ratio: float = 0.15,
    val_ratio: float = 0.1
) -> DatasetDict:
    """
    Prepare the general translation dataset with multiple language pairs.

    Strategy:
    - 40% ko↔en (bidirectional)
    - 40% ja↔ko (bidirectional)
    - 20% direct pairs (ko↔ja, en↔ja)
    - 10-15% augmented with autotext patterns

    Args:
        data_dir: Base directory containing aihub_merged datasets
        max_samples_per_pair: Max samples per language pair (None for all)
        autotext_augmentation_ratio: Ratio of examples to augment with autotext
        val_ratio: Validation split ratio

    Returns:
        DatasetDict with 'train' and 'validation' splits
    """
    print("\n=== Preparing General Translation Dataset ===\n")

    all_examples = []

    # Load and process each language pair
    pairs_config = [
        ("ko_en", "ko", "en", 0.20),  # 20% of total (bidirectional = 40%)
        ("ja_ko", "ja", "ko", 0.20),  # 20% of total (bidirectional = 40%)
        ("ko_ja", "ko", "ja", 0.10),  # 10% of total
        ("en_ko", "en", "ko", 0.10),  # 10% of total (different from ko_en content)
    ]

    for pair_dir, src_lang, tgt_lang, ratio in pairs_config:
        # Calculate samples for this pair
        samples = int(max_samples_per_pair * ratio) if max_samples_per_pair else None

        try:
            dataset = load_aihub_dataset(data_dir, pair_dir, "train", samples)

            # Add language tags
            dataset = dataset.map(
                lambda x: {
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang
                }
            )

            # Format for translation
            formatted = dataset.map(
                format_translation_batch,
                batched=True,
                remove_columns=dataset.column_names
            )

            all_examples.extend([{"text": text} for text in formatted["text"]])

            print(f"  Added {len(formatted)} examples from {pair_dir}")

        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue

    if not all_examples:
        raise ValueError("No datasets loaded! Check data_dir path.")

    print(f"\nTotal examples before augmentation: {len(all_examples)}")

    # Augment with autotext patterns
    num_augmented = int(len(all_examples) * autotext_augmentation_ratio)
    autotext_ids = list(range(1001, 1050))  # Use IDs 1001-1049

    print(f"Augmenting with {num_augmented} autotext examples...")
    all_examples = augment_with_autotext(
        all_examples,
        num_augmented,
        autotext_ids,
        positions=["start", "end", "random"]
    )

    print(f"Total examples after augmentation: {len(all_examples)}")

    # Shuffle
    random.shuffle(all_examples)

    # Create dataset
    full_dataset = Dataset.from_list(all_examples)

    # Split into train/val
    dataset_dict = full_dataset.train_test_split(
        test_size=val_ratio,
        shuffle=True,
        seed=42
    )

    # Rename 'test' to 'validation'
    dataset_dict = DatasetDict({
        'train': dataset_dict['train'],
        'validation': dataset_dict['test']
    })

    print(f"\n✓ Dataset prepared:")
    print(f"  Train: {len(dataset_dict['train'])} examples")
    print(f"  Validation: {len(dataset_dict['validation'])} examples")

    return dataset_dict


def save_prepared_dataset(dataset_dict: DatasetDict, output_dir: str):
    """
    Save prepared dataset to disk.

    Args:
        dataset_dict: Dataset dictionary to save
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset_dict.save_to_disk(str(output_path))
    print(f"\n✓ Saved dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare general translation dataset from aihub_merged"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/tikim/dataset/aihub_merged",
        help="Base directory containing aihub_merged datasets"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed/general_translation",
        help="Output directory for prepared dataset"
    )
    parser.add_argument(
        "--max_samples_per_pair",
        type=int,
        default=50000,
        help="Maximum samples per language pair (total will be ~200k with bidirectional)"
    )
    parser.add_argument(
        "--autotext_ratio",
        type=float,
        default=0.15,
        help="Ratio of examples to augment with autotext (default: 0.15)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )

    args = parser.parse_args()

    # Prepare dataset
    dataset_dict = prepare_general_translation_dataset(
        data_dir=args.data_dir,
        max_samples_per_pair=args.max_samples_per_pair,
        autotext_augmentation_ratio=args.autotext_ratio,
        val_ratio=args.val_ratio
    )

    # Save to disk
    save_prepared_dataset(dataset_dict, args.output)

    # Show samples
    print("\n=== Sample Examples ===")
    for i in range(min(3, len(dataset_dict['train']))):
        print(f"\nExample {i+1}:")
        print(dataset_dict['train'][i]['text'])


if __name__ == "__main__":
    main()
