"""
Dataset formatting utilities for translation training.

Formats parallel translation data into the model's expected format:
<src:lang><tgt:lang>
SOURCE_TEXT
###
TARGET_TEXT

Copyright 2025
Licensed under the Apache License, Version 2.0
"""

from typing import Dict, Optional
import random


# Separator between source and target
SEPLINE = "###"


def format_translation_example(
    source: str,
    target: str,
    src_lang: str,
    tgt_lang: str
) -> Dict[str, str]:
    """
    Format a single translation example.

    Args:
        source: Source text
        target: Target text
        src_lang: Source language code (ja, ko, en)
        tgt_lang: Target language code (ja, ko, en)

    Returns:
        Dictionary with 'text' field containing formatted example

    Example:
        >>> format_translation_example("Hello", "こんにちは", "en", "ja")
        {'text': '<src:en><tgt:ja>\\nHello\\n###\\nこんにちは'}
    """
    prefix = f"<src:{src_lang}><tgt:{tgt_lang}>"
    formatted = f"{prefix}\n{source.strip()}\n{SEPLINE}\n{target.strip()}"

    return {"text": formatted}


def format_translation_batch(examples: Dict) -> Dict[str, list]:
    """
    Format a batch of translation examples for dataset mapping.

    Expected input columns: sourceString, targetString, src_lang, tgt_lang

    Args:
        examples: Batch from dataset with multiple examples

    Returns:
        Dictionary with 'text' field containing list of formatted examples
    """
    texts = []

    for i in range(len(examples['sourceString'])):
        source = examples['sourceString'][i]
        target = examples['targetString'][i]
        src_lang = examples['src_lang'][i]
        tgt_lang = examples['tgt_lang'][i]

        formatted = format_translation_example(source, target, src_lang, tgt_lang)
        texts.append(formatted['text'])

    return {"text": texts}


def add_autotext_to_example(
    example: Dict[str, str],
    autotext_id: int,
    position: str = "start"
) -> Dict[str, str]:
    """
    Add autotext markers to a translation example for training.

    This helps the model learn to preserve autotext patterns.

    Args:
        example: Example with 'text' field
        autotext_id: Autotext ID to insert
        position: Where to insert ('start', 'end', 'random')

    Returns:
        Modified example with autotext markers
    """
    text = example['text']
    lines = text.split('\n')

    # Find source and target lines (skip prefix and separator)
    prefix_line = lines[0]
    source_line = lines[1]
    sep_line = lines[2]
    target_line = lines[3]

    autotext_marker = f"<<AT:{autotext_id}>>"

    # Add to both source and target (model learns to copy)
    if position == "start":
        source_line = autotext_marker + source_line
        target_line = autotext_marker + target_line
    elif position == "end":
        source_line = source_line + autotext_marker
        target_line = target_line + autotext_marker
    elif position == "random":
        # Insert at random position in text
        if random.random() < 0.5:
            source_line = autotext_marker + source_line
            target_line = autotext_marker + target_line
        else:
            source_line = source_line + autotext_marker
            target_line = target_line + autotext_marker

    # Reconstruct
    modified_text = f"{prefix_line}\n{source_line}\n{sep_line}\n{target_line}"
    example['text'] = modified_text

    return example


def create_bidirectional_examples(
    source: str,
    target: str,
    lang1: str,
    lang2: str
) -> list:
    """
    Create bidirectional translation examples from a parallel pair.

    Args:
        source: Text in lang1
        target: Text in lang2
        lang1: First language code
        lang2: Second language code

    Returns:
        List of two examples (lang1→lang2 and lang2→lang1)

    Example:
        >>> create_bidirectional_examples("Hello", "こんにちは", "en", "ja")
        [
            {'text': '<src:en><tgt:ja>\\nHello\\n###\\nこんにちは'},
            {'text': '<src:ja><tgt:en>\\nこんにちは\\n###\\nHello'}
        ]
    """
    examples = []

    # Direction 1: lang1 → lang2
    examples.append(format_translation_example(source, target, lang1, lang2))

    # Direction 2: lang2 → lang1
    examples.append(format_translation_example(target, source, lang2, lang1))

    return examples


def augment_with_autotext(
    examples: list,
    num_augmented: int,
    autotext_ids: list,
    positions: list = ["start", "end"]
) -> list:
    """
    Augment a list of examples with autotext patterns.

    Args:
        examples: List of example dictionaries
        num_augmented: Number of examples to augment
        autotext_ids: List of autotext IDs to use
        positions: List of positions to insert autotext

    Returns:
        Original examples + augmented examples
    """
    augmented = []

    for _ in range(num_augmented):
        # Pick random example, autotext ID, and position
        example = random.choice(examples).copy()
        autotext_id = random.choice(autotext_ids)
        position = random.choice(positions)

        # Add autotext
        augmented_example = add_autotext_to_example(example, autotext_id, position)
        augmented.append(augmented_example)

    return examples + augmented


if __name__ == "__main__":
    # Test formatting
    print("Testing dataset formatting...\n")

    # Test single example
    example = format_translation_example(
        "Hello, how are you?",
        "こんにちは、お元気ですか？",
        "en",
        "ja"
    )
    print("Single example:")
    print(example['text'])
    print()

    # Test bidirectional
    examples = create_bidirectional_examples(
        "Thank you very much",
        "どうもありがとうございます",
        "en",
        "ja"
    )
    print("Bidirectional examples:")
    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex['text']}")
        print()

    # Test autotext augmentation
    print("With autotext marker:")
    example_with_at = add_autotext_to_example(examples[0].copy(), 1023, "start")
    print(example_with_at['text'])
