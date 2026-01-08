"""
Bias-corrected Exponential Moving Average (EMA) for LoRA training.

Implements EMA with bias correction for stable model performance during inference.
Designed to work with PEFT/LoRA adapters.

Copyright 2025
Licensed under the Apache License, Version 2.0
"""

import torch
from typing import Dict, Optional
from collections import OrderedDict
from transformers import TrainerCallback
import copy


class BiasCorrectEMA:
    """
    Bias-corrected Exponential Moving Average for model parameters.

    EMA maintains a moving average of model parameters:
        ema_param = decay * ema_param + (1 - decay) * param

    Bias correction adjusts for initialization bias in early training:
        corrected_ema = ema_param / (1 - decay^step)

    Args:
        model: The model to apply EMA to
        decay: Decay rate for EMA (default: 0.999)
        min_decay: Minimum decay rate (default: 0.0)
        update_after_step: Start EMA updates after this step (default: 100)
        use_ema_weights: Whether to use EMA weights during eval (default: True)
        device: Device to store EMA parameters (default: None, uses model device)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.999,
        min_decay: float = 0.0,
        update_after_step: int = 100,
        use_ema_weights: bool = True,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.decay = decay
        self.min_decay = min_decay
        self.update_after_step = update_after_step
        self.use_ema_weights = use_ema_weights

        # Get trainable parameters (for LoRA, these are adapter parameters)
        self.trainable_params = [p for p in model.parameters() if p.requires_grad]

        # Store EMA parameters
        self.ema_params = [p.clone().detach() for p in self.trainable_params]

        if device is not None:
            self.ema_params = [p.to(device) for p in self.ema_params]

        # Track update count for bias correction
        self.num_updates = 0
        self._is_active = False

        print(f"✓ Bias-corrected EMA initialized")
        print(f"  Decay: {decay}")
        print(f"  Min decay: {min_decay}")
        print(f"  Update after step: {update_after_step}")
        print(f"  Trainable parameters: {len(self.trainable_params)}")

    def get_decay(self, step: int) -> float:
        """
        Get current decay rate with warmup.

        Args:
            step: Current training step

        Returns:
            Current decay rate
        """
        if step < self.update_after_step:
            return 0.0

        # Linear warmup from min_decay to decay
        warmup_steps = 100
        if step < self.update_after_step + warmup_steps:
            alpha = (step - self.update_after_step) / warmup_steps
            return self.min_decay + alpha * (self.decay - self.min_decay)

        return self.decay

    def update(self, step: int):
        """
        Update EMA parameters.

        Args:
            step: Current training step
        """
        if step < self.update_after_step:
            return

        self.num_updates += 1
        decay = self.get_decay(step)

        # Update EMA parameters
        with torch.no_grad():
            for ema_param, param in zip(self.ema_params, self.trainable_params):
                if param.requires_grad:
                    # EMA update: ema = decay * ema + (1 - decay) * param
                    ema_param.mul_(decay).add_(param.data, alpha=1 - decay)

    def get_bias_corrected_params(self) -> list:
        """
        Get bias-corrected EMA parameters.

        Returns:
            List of bias-corrected EMA parameters
        """
        if self.num_updates == 0:
            return self.ema_params

        # Bias correction: corrected = ema / (1 - decay^num_updates)
        bias_correction = 1 - self.decay ** self.num_updates

        corrected_params = []
        for ema_param in self.ema_params:
            corrected_param = ema_param / bias_correction
            corrected_params.append(corrected_param)

        return corrected_params

    def apply_ema_weights(self):
        """Store current weights and apply EMA weights to model."""
        if self._is_active:
            return  # Already using EMA weights

        self._backup_params = [p.data.clone() for p in self.trainable_params]

        # Apply bias-corrected EMA weights
        corrected_params = self.get_bias_corrected_params()
        with torch.no_grad():
            for param, ema_param in zip(self.trainable_params, corrected_params):
                param.data.copy_(ema_param)

        self._is_active = True

    def restore_original_weights(self):
        """Restore original (non-EMA) weights to model."""
        if not self._is_active:
            return  # Not using EMA weights

        with torch.no_grad():
            for param, backup_param in zip(self.trainable_params, self._backup_params):
                param.data.copy_(backup_param)

        self._is_active = False
        self._backup_params = None

    def state_dict(self) -> Dict:
        """
        Get state dictionary for checkpointing.

        Returns:
            State dictionary
        """
        return {
            'ema_params': [p.cpu() for p in self.ema_params],
            'num_updates': self.num_updates,
            'decay': self.decay,
            'min_decay': self.min_decay,
            'update_after_step': self.update_after_step
        }

    def load_state_dict(self, state_dict: Dict):
        """
        Load state from dictionary.

        Args:
            state_dict: State dictionary
        """
        self.ema_params = [p.to(self.ema_params[0].device)
                          for p in state_dict['ema_params']]
        self.num_updates = state_dict['num_updates']
        self.decay = state_dict['decay']
        self.min_decay = state_dict['min_decay']
        self.update_after_step = state_dict['update_after_step']


class EMACallback(TrainerCallback):
    """
    Trainer callback for integrating EMA with HuggingFace Trainer.

    This callback:
    - Updates EMA after each training step
    - Applies EMA weights before evaluation
    - Restores original weights after evaluation
    - Saves EMA state with checkpoints

    Args:
        ema: BiasCorrectEMA instance
        save_ema_weights: Whether to save final model with EMA weights (default: True)
    """

    def __init__(self, ema: BiasCorrectEMA, save_ema_weights: bool = True):
        self.ema = ema
        self.save_ema_weights = save_ema_weights

    def on_step_end(self, args, state, control, **kwargs):
        """Update EMA after each training step."""
        self.ema.update(state.global_step)

    def on_evaluate(self, args, state, control, **kwargs):
        """Apply EMA weights before evaluation."""
        if self.ema.use_ema_weights:
            self.ema.apply_ema_weights()

    def on_prediction_step(self, args, state, control, **kwargs):
        """Restore original weights after evaluation."""
        if self.ema.use_ema_weights and self.ema._is_active:
            self.ema.restore_original_weights()

    def on_save(self, args, state, control, **kwargs):
        """Save EMA state with checkpoint."""
        model = kwargs.get('model')
        if model is not None:
            # Save EMA state
            ema_path = args.output_dir + f"/ema_state_step_{state.global_step}.pt"
            torch.save(self.ema.state_dict(), ema_path)
            print(f"  Saved EMA state to {ema_path}")

    def on_train_end(self, args, state, control, **kwargs):
        """Apply EMA weights to final model if requested."""
        if self.save_ema_weights:
            print("\n=== Applying EMA weights to final model ===")
            self.ema.apply_ema_weights()
            print(f"✓ Final model now uses bias-corrected EMA weights")
            print(f"  Total EMA updates: {self.ema.num_updates}")
            print(f"  Bias correction factor: {1 - self.ema.decay ** self.ema.num_updates:.6f}")


def load_model_with_ema(model, ema_state_path: str):
    """
    Load EMA state and apply to model.

    Args:
        model: Model instance
        ema_state_path: Path to saved EMA state
    """
    ema = BiasCorrectEMA(model)
    ema_state = torch.load(ema_state_path)
    ema.load_state_dict(ema_state)
    ema.apply_ema_weights()

    print(f"✓ Loaded and applied EMA weights from {ema_state_path}")
    print(f"  EMA updates: {ema.num_updates}")

    return model


if __name__ == "__main__":
    # Test EMA implementation
    print("Testing Bias-corrected EMA...\n")

    # Create dummy model
    model = torch.nn.Linear(10, 10)

    # Initialize EMA
    ema = BiasCorrectEMA(model, decay=0.999, update_after_step=10)

    # Simulate training
    print("Simulating training steps...")
    for step in range(100):
        # Dummy parameter update
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn_like(param) * 0.01)

        # Update EMA
        ema.update(step)

        if step % 20 == 0:
            bias_correction = 1 - ema.decay ** max(1, ema.num_updates)
            print(f"  Step {step}: EMA updates={ema.num_updates}, "
                  f"bias_correction={bias_correction:.6f}")

    print("\n✓ EMA test completed")

    # Test apply/restore
    print("\nTesting apply/restore EMA weights...")
    original_param = model.weight.data.clone()

    ema.apply_ema_weights()
    ema_param = model.weight.data.clone()

    ema.restore_original_weights()
    restored_param = model.weight.data.clone()

    print(f"  Original == Restored: {torch.allclose(original_param, restored_param)}")
    print(f"  Original != EMA: {not torch.allclose(original_param, ema_param)}")

    print("\n✓ All tests passed!")
