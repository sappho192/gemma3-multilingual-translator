"""
Autotext normalization and restoration utilities.

These utilities handle the conversion between:
- <autotext>phrase</autotext> format (external representation)
- <<AT:id>> format (internal model representation)

Copyright 2025
Licensed under the Apache License, Version 2.0
"""

import re
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List


# Autotext markers
AT_OPEN = "<<AT:"
AT_CLOSE = ">>"
AT_RE = re.compile(r"<<AT:(\d+)>>")
AUTOTEXT_XML_RE = re.compile(r"<autotext>(.*?)</autotext>")


def normalize_phrase(s: str) -> str:
    """
    Normalize a phrase by:
    - Replacing full-width spaces with half-width
    - Collapsing multiple spaces into one
    - Stripping leading/trailing whitespace

    Args:
        s: Input phrase

    Returns:
        Normalized phrase
    """
    return re.sub(r"\s+", " ", s.replace("　", " ").strip())


def normalize_autotext(text: str, lang: str, to_id: Dict[Tuple[str, str], int]) -> str:
    """
    Convert <autotext>phrase</autotext> format to <<AT:id>> format.

    Args:
        text: Input text with <autotext> tags
        lang: Language code of the text (ja, ko, en)
        to_id: Mapping from (lang, normalized_phrase) to ID

    Returns:
        Text with <autotext> tags replaced by <<AT:id>>

    Example:
        >>> to_id = {("ja", "おはようございます"): 1023}
        >>> normalize_autotext("<autotext>おはようございます</autotext>", "ja", to_id)
        "<<AT:1023>>"
    """
    def replace_func(match):
        phrase = match.group(1)
        phrase_norm = normalize_phrase(phrase)
        id_ = to_id.get((lang, phrase_norm))

        if id_ is not None:
            return f"{AT_OPEN}{id_}{AT_CLOSE}"
        else:
            # If no mapping found, keep original
            return match.group(0)

    return AUTOTEXT_XML_RE.sub(replace_func, text)


def restore_autotext(text: str, ui_lang: str, from_id: Dict[Tuple[str, int], str]) -> str:
    """
    Convert <<AT:id>> format back to phrases in the target UI language.

    Args:
        text: Text with <<AT:id>> markers
        ui_lang: UI language code (ja, ko, en) for restoration
        from_id: Mapping from (ui_lang, id) to phrase

    Returns:
        Text with <<AT:id>> markers replaced by phrases

    Example:
        >>> from_id = {("en", 1023): "Good morning"}
        >>> restore_autotext("<<AT:1023>>", "en", from_id)
        "Good morning"
    """
    def replace_func(match):
        id_ = int(match.group(1))
        phrase = from_id.get((ui_lang, id_))

        if phrase is not None:
            return phrase
        else:
            # If no mapping found, keep the marker
            return match.group(0)

    return AT_RE.sub(replace_func, text)


def load_autotext_mappings(mapping_file: str) -> Tuple[Dict[Tuple[str, str], int], Dict[Tuple[str, int], str]]:
    """
    Load autotext mappings from JSON file.

    Expected format:
    {
        "mappings": [
            {
                "id": 1023,
                "phrases": {
                    "ja": "おはようございます",
                    "ko": "안녕하세요",
                    "en": "Good morning"
                }
            },
            ...
        ]
    }

    Args:
        mapping_file: Path to JSON mapping file

    Returns:
        Tuple of (to_id_dict, from_id_dict)
    """
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    to_id = {}    # (lang, phrase) -> id
    from_id = {}  # (lang, id) -> phrase

    for entry in data.get("mappings", []):
        id_ = entry["id"]
        phrases = entry["phrases"]

        for lang, phrase in phrases.items():
            phrase_norm = normalize_phrase(phrase)
            to_id[(lang, phrase_norm)] = id_
            from_id[(lang, id_)] = phrase

    return to_id, from_id


def save_autotext_mappings(
    mappings: list,
    output_file: str
):
    """
    Save autotext mappings to JSON file.

    Args:
        mappings: List of mapping entries with id and phrases
        output_file: Output JSON file path
    """
    data = {"mappings": mappings}

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(mappings)} autotext mappings to {output_file}")


def create_sample_mappings(output_file: str = "./data/autotext_mappings/sample.json"):
    """
    Create sample autotext mappings for testing.

    Args:
        output_file: Output file path
    """
    sample_mappings = [
        {
            "id": 1001,
            "phrases": {
                "ja": "こんにちは",
                "ko": "안녕하세요",
                "en": "Hello"
            }
        },
        {
            "id": 1002,
            "phrases": {
                "ja": "ありがとう",
                "ko": "감사합니다",
                "en": "Thank you"
            }
        },
        {
            "id": 1003,
            "phrases": {
                "ja": "さようなら",
                "ko": "안녕히 가세요",
                "en": "Goodbye"
            }
        },
        {
            "id": 1023,
            "phrases": {
                "ja": "おはようございます",
                "ko": "좋은 아침입니다",
                "en": "Good morning"
            }
        }
    ]

    save_autotext_mappings(sample_mappings, output_file)


def validate_autotext_pattern(text: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that all <<AT:id>> patterns in text are well-formed.

    Args:
        text: Text to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Find all potential AT patterns
    matches = re.findall(r"<<AT:[^>]*>>", text)

    for match in matches:
        if not AT_RE.match(match):
            return False, f"Malformed autotext pattern: {match}"

    # Check for unclosed patterns
    open_count = text.count(AT_OPEN)
    close_count = text.count(AT_CLOSE)

    if open_count != close_count:
        return False, f"Mismatched autotext markers: {open_count} opens, {close_count} closes"

    return True, None


if __name__ == "__main__":
    # Create sample mappings for testing
    print("Creating sample autotext mappings...")
    create_sample_mappings()

    # Test the utilities
    print("\nTesting autotext utilities...")

    to_id, from_id = load_autotext_mappings("./data/autotext_mappings/sample.json")

    test_text = "<autotext>おはようございます</autotext>久しぶり〜"
    print(f"\nOriginal: {test_text}")

    normalized = normalize_autotext(test_text, "ja", to_id)
    print(f"Normalized: {normalized}")

    restored_en = restore_autotext(normalized, "en", from_id)
    print(f"Restored (en): {restored_en}")

    restored_ko = restore_autotext(normalized, "ko", from_id)
    print(f"Restored (ko): {restored_ko}")

    # Validate patterns
    is_valid, error = validate_autotext_pattern(normalized)
    print(f"\nValidation: {'✓ Valid' if is_valid else f'✗ Invalid - {error}'}")
