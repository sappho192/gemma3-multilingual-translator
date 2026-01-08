"""
Tokenizer wrapper for Gemma3 translation models.
Supports both HuggingFace tokenizers and raw tokenizers library.
"""

import os
import json
from typing import List, Union, Dict
import numpy as np

try:
    from tokenizers import Tokenizer
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False


class GemmaTokenizer:
    """
    Lightweight tokenizer wrapper for Gemma3 models.
    Optimized for translation task without PyTorch dependency.
    """

    # Language codes for translation
    LANG_CODES = {"ko", "en", "ja"}

    def __init__(self, model_dir: str):
        """
        Initialize the tokenizer.

        Args:
            model_dir: Directory containing tokenizer files
        """
        self.model_dir = model_dir

        # Try to load using transformers (preferred for chat template support)
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self._use_hf_tokenizer = True
        except ImportError:
            # Fallback to tokenizers library
            if not TOKENIZERS_AVAILABLE:
                raise ImportError("Either transformers or tokenizers library is required.")
            tokenizer_path = os.path.join(model_dir, "tokenizer.json")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
            self.tokenizer = Tokenizer.from_file(tokenizer_path)
            self._use_hf_tokenizer = False

        # Special tokens from Gemma3 config
        self.bos_token_id = 2
        self.eos_token_ids = [1, 106]  # Multiple EOS tokens
        self.pad_token_id = 0

        # Get vocab size
        if self._use_hf_tokenizer:
            self.vocab_size = len(self.tokenizer)
        else:
            self.vocab_size = self.tokenizer.get_vocab_size()

    def format_translation_prompt(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str
    ) -> str:
        """
        Format input text for translation.

        Args:
            text: Source text to translate
            src_lang: Source language code (ko, en, ja)
            tgt_lang: Target language code (ko, en, ja)

        Returns:
            Formatted prompt string
        """
        if src_lang not in self.LANG_CODES:
            raise ValueError(f"Source language must be one of: {self.LANG_CODES}")
        if tgt_lang not in self.LANG_CODES:
            raise ValueError(f"Target language must be one of: {self.LANG_CODES}")

        return f"<src:{src_lang}><tgt:{tgt_lang}>\n{text}\n###\n"

    def encode(self, text: str, add_special_tokens: bool = True) -> np.ndarray:
        """
        Encode text to token IDs.

        Args:
            text: Input text to tokenize
            add_special_tokens: Whether to add BOS token

        Returns:
            numpy array of token IDs (int64)
        """
        if self._use_hf_tokenizer:
            token_ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return np.array(token_ids, dtype=np.int64)
        else:
            encoding = self.tokenizer.encode(text, add_special_tokens=False)
            token_ids = list(encoding.ids)

            if add_special_tokens:
                token_ids = [self.bos_token_id] + token_ids

            return np.array(token_ids, dtype=np.int64)

    def decode(self, token_ids: Union[List[int], np.ndarray], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()

        if self._use_hf_tokenizer:
            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            if skip_special_tokens:
                special_tokens = {self.bos_token_id, self.pad_token_id} | set(self.eos_token_ids)
                token_ids = [tid for tid in token_ids if tid not in special_tokens]

            return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def extract_translation(self, output: str) -> str:
        """
        Extract translated text from model output.

        Args:
            output: Full decoded output from model

        Returns:
            Extracted translation (text after ### delimiter)
        """
        if "###" in output:
            parts = output.split("###")
            if len(parts) > 1:
                return parts[-1].strip()
        return output.strip()

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
