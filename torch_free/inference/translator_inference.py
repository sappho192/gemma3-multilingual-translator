"""
Main inference class for Gemma3 Multilingual Translator.
PyTorch-free implementation for translation.
"""

import json
import os
from typing import Optional, List, Tuple
import numpy as np

from .gemma_tokenizer import GemmaTokenizer
from .gemma_session import Gemma3Session
from .kv_cache import KVCacheManager
from .generation import GenerationMixin


class TranslatorInferencer(GenerationMixin):
    """
    Complete inference pipeline for Gemma3 Multilingual Translator.
    Handles tokenization, ONNX inference, and text generation without PyTorch.
    """

    # Supported language pairs
    LANG_CODES = {"ko", "en", "ja"}

    def __init__(self, model_dir: str, precision: str = "fp32", verbose: bool = True):
        """
        Initialize translator inferencer.

        Args:
            model_dir: Directory containing ONNX models and tokenizer
            precision: Model precision ('fp32', 'fp16', 'q4', 'q4f16')
            verbose: Whether to print initialization info
        """
        self.model_dir = model_dir
        self.precision = precision
        self.verbose = verbose

        if self.verbose:
            print(f"Initializing Translator with precision: {precision}")

        # Load tokenizer
        self.tokenizer = GemmaTokenizer(model_dir)
        if self.verbose:
            print(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")

        # Load ONNX session
        self.session = Gemma3Session(model_dir, precision)

        # Initialize KV cache manager
        self.kv_cache_manager = KVCacheManager(
            num_layers=self.session.num_layers,
            num_kv_heads=self.session.num_kv_heads,
            head_dim=self.session.head_dim,
            dtype=self.session.kv_cache_dtype
        )

        # Load generation config
        self.generation_config = self._load_generation_config()

        if self.verbose:
            print("Initialization complete!")

    def _load_generation_config(self) -> dict:
        """Load generation configuration."""
        config_path = os.path.join(self.model_dir, "generation_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
        return {}

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """
        Translate text from source language to target language.

        Args:
            text: Source text to translate
            src_lang: Source language code ('ko', 'en', 'ja')
            tgt_lang: Target language code ('ko', 'en', 'ja')
            max_new_tokens: Maximum number of tokens to generate
            do_sample: Whether to use sampling (False = greedy)
            temperature: Sampling temperature
            top_k: Top-k filtering parameter
            top_p: Top-p filtering parameter

        Returns:
            Translated text
        """
        # Validate language codes
        if src_lang not in self.LANG_CODES:
            raise ValueError(f"Source language must be one of: {self.LANG_CODES}")
        if tgt_lang not in self.LANG_CODES:
            raise ValueError(f"Target language must be one of: {self.LANG_CODES}")

        # Format prompt
        prompt = self.tokenizer.format_translation_prompt(text, src_lang, tgt_lang)

        if self.verbose:
            print(f"\nTranslating: {text}")
            print(f"Direction: {src_lang} -> {tgt_lang}")

        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        if self.verbose:
            print(f"Input tokens: {len(input_ids)}")

        # Generate tokens
        generated_ids = self._generate_tokens(
            input_ids,
            max_length=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

        # Decode full output
        full_output = self.tokenizer.decode(
            np.concatenate([input_ids, generated_ids]),
            skip_special_tokens=True
        )

        # Extract translation
        translation = self.tokenizer.extract_translation(full_output)

        if self.verbose:
            print(f"Translation: {translation}")

        return translation

    def _generate_tokens(
        self,
        input_ids: np.ndarray,
        max_length: int,
        do_sample: bool,
        temperature: float,
        top_k: int,
        top_p: float
    ) -> np.ndarray:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs [seq_len]
            max_length: Maximum tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter

        Returns:
            Array of generated token IDs (excluding prompt)
        """
        batch_size = 1
        input_ids = input_ids.reshape(1, -1)  # [1, seq_len]

        # Initialize KV cache
        kv_cache = self.kv_cache_manager.create_empty_cache(batch_size)

        # Create attention mask and position IDs for prompt
        seq_len = input_ids.shape[1]
        attention_mask = self.create_attention_mask(seq_len, 0, batch_size)
        position_ids = self.create_position_ids(seq_len, 0, batch_size)

        # Process prompt (prefill)
        logits, present_kv = self.session.run(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache
        )

        # Update cache
        kv_cache = self.kv_cache_manager.update_cache({}, present_kv)

        # Get last token logits and select next token
        last_logits = logits[0, -1, :]  # [vocab_size]

        if do_sample:
            next_token = self.sample_with_temperature(
                last_logits, temperature, top_k, top_p
            )
        else:
            next_token = self.greedy_decode(last_logits)

        generated_tokens = [next_token]

        # Autoregressive generation loop
        eos_token_ids = self.tokenizer.eos_token_ids
        past_len = seq_len

        for _ in range(max_length - 1):
            # Check for EOS
            if self.is_eos_token(next_token, eos_token_ids):
                break

            # Prepare input for next step
            next_input_ids = np.array([[next_token]], dtype=np.int64)  # [1, 1]

            # Update attention mask and position IDs
            attention_mask = self.create_attention_mask(1, past_len, batch_size)
            position_ids = self.create_position_ids(1, past_len, batch_size)

            # Run model
            logits, present_kv = self.session.run(
                input_ids=next_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )

            # Update cache
            kv_cache = self.kv_cache_manager.update_cache(kv_cache, present_kv)
            past_len += 1

            # Select next token
            last_logits = logits[0, -1, :]

            if do_sample:
                next_token = self.sample_with_temperature(
                    last_logits, temperature, top_k, top_p
                )
            else:
                next_token = self.greedy_decode(last_logits)

            generated_tokens.append(next_token)

        return np.array(generated_tokens, dtype=np.int64)

    def batch_translate(
        self,
        texts: List[str],
        src_lang: str,
        tgt_lang: str,
        **kwargs
    ) -> List[str]:
        """
        Translate multiple texts (sequential processing).

        Args:
            texts: List of source texts
            src_lang: Source language code
            tgt_lang: Target language code
            **kwargs: Additional generation arguments

        Returns:
            List of translated texts
        """
        return [
            self.translate(text, src_lang, tgt_lang, **kwargs)
            for text in texts
        ]

    def translate_interactive(self):
        """
        Interactive translation mode.
        """
        print("\n" + "=" * 60)
        print("Gemma3 Multilingual Translator - Interactive Mode")
        print("=" * 60)
        print("Supported languages: ko (Korean), en (English), ja (Japanese)")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60 + "\n")

        while True:
            try:
                # Get source language
                src_lang = input("Source language (ko/en/ja): ").strip().lower()
                if src_lang in ("quit", "exit"):
                    break
                if src_lang not in self.LANG_CODES:
                    print(f"Invalid language. Use one of: {self.LANG_CODES}")
                    continue

                # Get target language
                tgt_lang = input("Target language (ko/en/ja): ").strip().lower()
                if tgt_lang in ("quit", "exit"):
                    break
                if tgt_lang not in self.LANG_CODES:
                    print(f"Invalid language. Use one of: {self.LANG_CODES}")
                    continue

                # Get text to translate
                text = input("Text to translate: ").strip()
                if text.lower() in ("quit", "exit"):
                    break
                if not text:
                    continue

                # Translate
                result = self.translate(text, src_lang, tgt_lang)
                print(f"\n>>> {result}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
                continue

        print("\nGoodbye!")
