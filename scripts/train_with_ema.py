"""
Train translation LoRA with Bias-corrected Exponential Moving Average (EMA).

This script extends the simple translation training with EMA for improved
model stability and generalization.

Copyright 2025
Licensed under the Apache License, Version 2.0
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
from huggingface_hub import login
from dotenv import load_dotenv

from lora_config import get_lora_config, get_bnb_config, print_lora_config
from ema_utils import BiasCorrectEMA, EMACallback


def setup_hf_auth():
    """Set up Hugging Face authentication"""
    load_dotenv()
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        print("Warning: HF_TOKEN not found in .env file")
    else:
        login(hf_token)
        print("✓ Logged in to Hugging Face")


def load_prepared_dataset(dataset_path: str):
    """
    Load prepared dataset from disk.

    Args:
        dataset_path: Path to prepared dataset directory

    Returns:
        DatasetDict with train and validation splits
    """
    print(f"\nLoading prepared dataset from {dataset_path}...")
    dataset = load_from_disk(dataset_path)

    print(f"✓ Dataset loaded:")
    print(f"  Train: {len(dataset['train']):,} examples")
    print(f"  Validation: {len(dataset['validation']):,} examples")

    return dataset


def train_with_ema(
    base_model_name: str,
    dataset_path: str,
    output_path: str,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_length: int = 512,
    gradient_checkpointing: bool = True,
    save_strategy: str = "epoch",
    gradient_accumulation_steps: int = 4,
    # EMA parameters
    ema_decay: float = 0.999,
    ema_min_decay: float = 0.0,
    ema_update_after_step: int = 100,
    use_ema_weights: bool = True,
    save_ema_weights: bool = True,
    # Resume parameter
    resume_from_checkpoint: str = None
):
    """
    Train translation LoRA adapter with EMA.

    Args:
        base_model_name: Hugging Face model name (e.g., google/gemma-3-270m)
        dataset_path: Path to prepared dataset
        output_path: Output path for LoRA adapters
        num_epochs: Number of training epochs
        batch_size: Per-device batch size
        learning_rate: Learning rate
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        lora_dropout: LoRA dropout
        max_seq_length: Maximum sequence length
        gradient_checkpointing: Use gradient checkpointing
        save_strategy: Save strategy (epoch, steps)
        gradient_accumulation_steps: Gradient accumulation steps
        ema_decay: EMA decay rate (default: 0.999)
        ema_min_decay: Minimum EMA decay (default: 0.0)
        ema_update_after_step: Start EMA after this step (default: 100)
        use_ema_weights: Use EMA weights for eval (default: True)
        save_ema_weights: Save final model with EMA weights (default: True)
        resume_from_checkpoint: Path to checkpoint to resume from (default: None)
    """
    print("\n" + "="*60)
    print("TRAINING TRANSLATION LORA WITH BIAS-CORRECTED EMA")
    print("="*60)

    # Load dataset
    dataset = load_prepared_dataset(dataset_path)

    # Load tokenizer
    print(f"\nLoading tokenizer from {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")

    # Get configurations
    lora_config = get_lora_config(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_modules_to_save=False
    )
    print_lora_config(lora_config)

    bnb_config = get_bnb_config()

    # Load base model with quantization
    print(f"\nLoading base model: {base_model_name}")
    print("  (This may take a few minutes...)")

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation='eager',
        torch_dtype=torch.bfloat16
    )

    # Set pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        base_model.config.pad_token_id = tokenizer.eos_token_id

    print(f"✓ Model loaded")
    print(f"  Model dtype: {base_model.dtype}")
    print(f"  Device map: {base_model.hf_device_map}")

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=50,
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_total_limit=3,
        max_length=max_seq_length,
        gradient_checkpointing=gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if gradient_checkpointing else None,
        packing=False,
        optim="adamw_torch_fused",
        report_to="tensorboard",
        weight_decay=0.01,
        bf16=True,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        dataset_text_field="text",
        dataset_kwargs={"skip_prepare_dataset": False}
    )

    print("\n=== Training Configuration ===")
    print(f"  Base model: {base_model_name}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max sequence length: {max_seq_length}")
    print(f"  Gradient checkpointing: {gradient_checkpointing}")
    print(f"  Output directory: {output_path}")

    print("\n=== EMA Configuration ===")
    print(f"  EMA decay: {ema_decay}")
    print(f"  EMA min decay: {ema_min_decay}")
    print(f"  Update after step: {ema_update_after_step}")
    print(f"  Use EMA for evaluation: {use_ema_weights}")
    print(f"  Save final model with EMA: {save_ema_weights}")

    # Initialize trainer
    print("\n=== Initializing Trainer ===")

    trainer = SFTTrainer(
        model=base_model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        peft_config=lora_config,
        processing_class=tokenizer
    )

    print("✓ Trainer initialized")

    # Initialize EMA after PEFT model is created
    print("\n=== Initializing EMA ===")
    ema = BiasCorrectEMA(
        model=trainer.model,
        decay=ema_decay,
        min_decay=ema_min_decay,
        update_after_step=ema_update_after_step,
        use_ema_weights=use_ema_weights
    )

    # Add EMA callback
    ema_callback = EMACallback(ema, save_ema_weights=save_ema_weights)
    trainer.add_callback(ema_callback)

    print("✓ EMA callback added to trainer")

    # Train
    if resume_from_checkpoint:
        print(f"\n=== Resuming Training from {resume_from_checkpoint} ===\n")
    else:
        print("\n=== Starting Training ===\n")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save final adapters
    print(f"\n=== Saving LoRA Adapters ===")
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)

    # Save final EMA state
    final_ema_path = Path(output_path) / "ema_state_final.pt"
    torch.save(ema.state_dict(), str(final_ema_path))
    print(f"✓ Final EMA state saved to {final_ema_path}")

    print(f"✓ LoRA adapters saved to {output_path}")

    # Plot training results
    plot_training_results(trainer, output_path)

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nLoRA adapters saved to: {output_path}")
    print(f"Training plot saved to: {output_path}/training_loss.png")
    print(f"TensorBoard logs: {output_path}/runs")
    print(f"EMA state: {final_ema_path}")
    if save_ema_weights:
        print(f"\n✓ Final model uses bias-corrected EMA weights")


def plot_training_results(trainer, output_path: str):
    """
    Plot and save training/validation loss curves.

    Args:
        trainer: Trained SFTTrainer instance
        output_path: Directory to save plot
    """
    print("\n=== Plotting Training Results ===")

    log_history = trainer.state.log_history

    # Extract losses
    train_losses = [log["loss"] for log in log_history if "loss" in log]
    epoch_train = [log["epoch"] for log in log_history if "loss" in log]
    eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_train, train_losses, label="Training Loss", marker='o', alpha=0.7)
    plt.plot(epoch_eval, eval_losses, label="Validation Loss (EMA)", marker='s', linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Translation LoRA with EMA - Training Progress")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save
    plot_path = Path(output_path) / "training_loss.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training plot saved to {plot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train translation LoRA with Bias-corrected EMA"
    )

    # Paths
    parser.add_argument(
        "--base_model",
        type=str,
        default="google/gemma-3-270m",
        help="Base model name or path (default: google/gemma-3-270m)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="./data/processed/simple_translation",
        help="Path to prepared dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./models/adapters/translator-with-ema",
        help="Output directory for LoRA adapters"
    )

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Max sequence length")

    # LoRA parameters
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")

    # EMA parameters
    parser.add_argument("--ema_decay", type=float, default=0.999,
                       help="EMA decay rate (default: 0.999)")
    parser.add_argument("--ema_min_decay", type=float, default=0.0,
                       help="Minimum EMA decay (default: 0.0)")
    parser.add_argument("--ema_update_after_step", type=int, default=100,
                       help="Start EMA updates after this step (default: 100)")
    parser.add_argument("--no_ema_eval", action="store_true",
                       help="Don't use EMA weights for evaluation")
    parser.add_argument("--no_save_ema", action="store_true",
                       help="Don't save final model with EMA weights")

    # Other options
    parser.add_argument("--no_gradient_checkpointing", action="store_true",
                       help="Disable gradient checkpointing (uses more memory)")
    parser.add_argument("--save_strategy", type=str, default="epoch",
                       choices=["epoch", "steps"], help="Save strategy")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    # Setup
    setup_hf_auth()

    # Train
    train_with_ema(
        base_model_name=args.base_model,
        dataset_path=args.dataset,
        output_path=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        save_strategy=args.save_strategy,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        ema_decay=args.ema_decay,
        ema_min_decay=args.ema_min_decay,
        ema_update_after_step=args.ema_update_after_step,
        use_ema_weights=not args.no_ema_eval,
        save_ema_weights=not args.no_save_ema,
        resume_from_checkpoint=args.resume_from_checkpoint
    )


if __name__ == "__main__":
    main()
