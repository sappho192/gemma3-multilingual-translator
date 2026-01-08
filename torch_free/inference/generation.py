"""
Text generation algorithms for Gemma3 models.
Implements greedy decoding and sampling strategies.
"""

import numpy as np
from typing import List


class GenerationMixin:
    """
    Mixin class providing text generation algorithms.
    """

    @staticmethod
    def greedy_decode(logits: np.ndarray) -> int:
        """
        Greedy decoding: select token with highest probability.

        Args:
            logits: Logits array, shape [vocab_size] or [1, 1, vocab_size]

        Returns:
            Token ID with highest score
        """
        if logits.ndim > 1:
            logits = logits.reshape(-1)
        return int(np.argmax(logits))

    @staticmethod
    def sample_top_k(logits: np.ndarray, top_k: int = 50) -> np.ndarray:
        """
        Apply top-k filtering to logits.

        Args:
            logits: Logits array, shape [vocab_size]
            top_k: Number of top tokens to keep

        Returns:
            Filtered logits with only top-k values
        """
        if top_k <= 0:
            return logits

        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < np.partition(logits, -top_k)[-top_k]
        logits[indices_to_remove] = -np.inf

        return logits

    @staticmethod
    def sample_top_p(logits: np.ndarray, top_p: float = 0.95) -> np.ndarray:
        """
        Apply nucleus (top-p) filtering to logits.

        Args:
            logits: Logits array, shape [vocab_size]
            top_p: Cumulative probability threshold

        Returns:
            Filtered logits with nucleus sampling
        """
        if top_p >= 1.0:
            return logits

        sorted_indices = np.argsort(logits)[::-1]
        sorted_logits = logits[sorted_indices]

        sorted_probs = np.exp(sorted_logits - np.max(sorted_logits))
        sorted_probs = sorted_probs / np.sum(sorted_probs)

        cumulative_probs = np.cumsum(sorted_probs)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -np.inf

        return logits

    @staticmethod
    def sample_with_temperature(
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 64,
        top_p: float = 0.95
    ) -> int:
        """
        Sample token with temperature scaling, top-k, and top-p filtering.

        Args:
            logits: Logits array, shape [vocab_size] or [1, 1, vocab_size]
            temperature: Temperature for scaling (higher = more random)
            top_k: Top-k filtering parameter
            top_p: Top-p (nucleus) filtering parameter

        Returns:
            Sampled token ID
        """
        if logits.ndim > 1:
            logits = logits.reshape(-1).copy()
        else:
            logits = logits.copy()

        if temperature != 1.0:
            logits = logits / temperature

        if top_k > 0:
            logits = GenerationMixin.sample_top_k(logits, top_k)

        if top_p < 1.0:
            logits = GenerationMixin.sample_top_p(logits, top_p)

        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)

        if not np.isfinite(probs).all():
            probs = np.ones_like(probs)
            probs = probs / np.sum(probs)

        token_id = np.random.choice(len(probs), p=probs)

        return int(token_id)

    @staticmethod
    def is_eos_token(token_id: int, eos_token_ids: List[int]) -> bool:
        """
        Check if token is an EOS token.

        Args:
            token_id: Token ID to check
            eos_token_ids: List of EOS token IDs

        Returns:
            True if token is EOS
        """
        return token_id in eos_token_ids

    @staticmethod
    def create_attention_mask(seq_len: int, past_len: int, batch_size: int = 1) -> np.ndarray:
        """
        Create attention mask for current step.

        Args:
            seq_len: Current sequence length (typically 1 during generation)
            past_len: Length of past KV cache
            batch_size: Batch size

        Returns:
            Attention mask of shape [batch_size, total_seq_len]
        """
        total_len = seq_len + past_len
        return np.ones((batch_size, total_len), dtype=np.int64)

    @staticmethod
    def create_position_ids(seq_len: int, past_len: int, batch_size: int = 1) -> np.ndarray:
        """
        Create position IDs for current step.

        Args:
            seq_len: Current sequence length
            past_len: Length of past KV cache
            batch_size: Batch size

        Returns:
            Position IDs of shape [batch_size, seq_len]
        """
        position_ids = np.arange(past_len, past_len + seq_len, dtype=np.int64)
        return np.tile(position_ids, (batch_size, 1))
