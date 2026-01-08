"""
KV cache management for Gemma3 autoregressive generation.
"""

import numpy as np
from typing import Dict, List


class KVCacheManager:
    """
    Manages key-value cache for efficient autoregressive generation.
    Handles cache initialization and updates for Gemma3's architecture.
    """

    def __init__(
        self,
        num_layers: int = 18,
        num_kv_heads: int = 1,
        head_dim: int = 256,
        dtype: str = "float32"
    ):
        """
        Initialize KV cache manager.

        Args:
            num_layers: Number of decoder layers (18 for Gemma3-270M)
            num_kv_heads: Number of key-value heads (1 for GQA)
            head_dim: Dimension of each attention head (256)
            dtype: Data type for cache ('float32' or 'float16')
        """
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = np.float16 if dtype == "float16" else np.float32

    def create_empty_cache(self, batch_size: int = 1) -> Dict[str, np.ndarray]:
        """
        Create empty KV cache for the first generation step.

        Args:
            batch_size: Batch size (typically 1 for inference)

        Returns:
            Dictionary mapping cache names to empty numpy arrays
        """
        cache = {}

        # Initialize empty cache for each layer
        # Shape: [batch_size, num_kv_heads, 0, head_dim]
        for i in range(self.num_layers):
            cache[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, self.num_kv_heads, 0, self.head_dim),
                dtype=self.dtype
            )
            cache[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, self.num_kv_heads, 0, self.head_dim),
                dtype=self.dtype
            )

        return cache

    def update_cache(
        self,
        past_cache: Dict[str, np.ndarray],
        present_outputs: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Update cache with new key-value pairs from model output.

        Args:
            past_cache: Previous KV cache dictionary
            present_outputs: List of present key/value tensors from model output

        Returns:
            Updated cache dictionary for next generation step
        """
        new_cache = {}

        # present_outputs contains [present.0.key, present.0.value, present.1.key, ...]
        for i in range(self.num_layers):
            key_idx = i * 2  # Even indices are keys
            value_idx = i * 2 + 1  # Odd indices are values

            new_cache[f"past_key_values.{i}.key"] = present_outputs[key_idx]
            new_cache[f"past_key_values.{i}.value"] = present_outputs[value_idx]

        return new_cache

    def get_cache_length(self, cache: Dict[str, np.ndarray]) -> int:
        """
        Get the current sequence length stored in cache.

        Args:
            cache: KV cache dictionary

        Returns:
            Length of cached sequence
        """
        first_key = "past_key_values.0.key"
        if first_key in cache:
            return cache[first_key].shape[2]  # sequence_length is dimension 2
        return 0
