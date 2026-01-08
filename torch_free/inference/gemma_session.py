"""
ONNX Runtime session manager for Gemma3 models.
"""

import os
import json
from typing import Dict, List, Tuple
import numpy as np

try:
    from onnxruntime import InferenceSession, SessionOptions
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class Gemma3Session:
    """
    Manages ONNX Runtime session for Gemma3 model inference.
    Supports multiple precision formats: fp32, fp16, q4, q4f16
    """

    def __init__(self, model_dir: str, precision: str = "fp32"):
        """
        Initialize ONNX session for Gemma3 model.

        Args:
            model_dir: Directory containing ONNX models
            precision: Model precision ('fp32', 'fp16', 'q4', 'q4f16')
        """
        if not ONNX_AVAILABLE:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

        self.model_dir = model_dir
        self.precision = precision

        # Load config
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Model parameters
        self.num_layers = self.config["num_hidden_layers"]
        self.num_kv_heads = self.config["num_key_value_heads"]
        self.head_dim = self.config["head_dim"]
        self.vocab_size = self.config["vocab_size"]

        # Determine KV cache dtype based on precision
        self.kv_cache_dtype = self._get_kv_cache_dtype()

        # Load ONNX model
        self.session = self._load_session()

    def _get_kv_cache_dtype(self) -> str:
        """
        Determine the KV cache data type based on model precision.

        Returns:
            'float16' or 'float32'
        """
        if "transformers.js_config" in self.config:
            kv_config = self.config["transformers.js_config"].get("kv_cache_dtype", {})
            if self.precision in kv_config:
                return kv_config[self.precision]

        if self.precision in ["fp16", "q4f16"]:
            return "float16"
        return "float32"

    def _load_session(self) -> "InferenceSession":
        """
        Load ONNX model and create inference session.

        Returns:
            InferenceSession object
        """
        model_files = {
            "fp32": "model.onnx",
            "fp16": "model_fp16.onnx",
            "q4": "model_q4.onnx",
            "q4f16": "model_q4f16.onnx"
        }

        if self.precision not in model_files:
            raise ValueError(
                f"Invalid precision '{self.precision}'. "
                f"Must be one of: {list(model_files.keys())}"
            )

        model_path = os.path.join(self.model_dir, "onnx", model_files[self.precision])

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        sess_options = SessionOptions()
        sess_options.log_severity_level = 3

        print(f"Loading ONNX model: {model_path}")
        session = InferenceSession(model_path, sess_options)

        return session

    def run(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        position_ids: np.ndarray,
        kv_cache: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Run inference on the model.

        Args:
            input_ids: Input token IDs, shape [batch_size, seq_len]
            attention_mask: Attention mask, shape [batch_size, total_seq_len]
            position_ids: Position IDs, shape [batch_size, seq_len]
            kv_cache: Key-value cache dictionary

        Returns:
            Tuple of (logits, present_kv_cache)
        """
        inputs = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
            "position_ids": position_ids.astype(np.int64),
        }

        inputs.update(kv_cache)

        outputs = self.session.run(None, inputs)

        logits = outputs[0]
        present_kv = outputs[1:]

        return logits, present_kv

    def get_input_names(self) -> List[str]:
        """Get list of input names."""
        return [inp.name for inp in self.session.get_inputs()]

    def get_output_names(self) -> List[str]:
        """Get list of output names."""
        return [out.name for out in self.session.get_outputs()]
