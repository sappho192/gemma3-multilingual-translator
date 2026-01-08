"""
PyTorch-free inference for Gemma3 Multilingual Translator.

This package provides ONNX Runtime-based inference without PyTorch dependency.
"""

from .inference import TranslatorInferencer

__all__ = ["TranslatorInferencer"]
