"""
Training script for FrawdLLM.

Usage:
    uv run python scripts/train.py                    # Train tiny model on subset
    uv run python scripts/train.py --config small    # Train small model
    uv run python scripts/train.py --full-data       # Use full TinyStories dataset
    uv run python scripts/train.py --wandb           # Enable W&B logging
"""

import argparse
from pathlib import Path

import torch

from src.model.gpt import FrawdLLM
from src.model.config import get_config
from src.training.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.fetch_data.download import get_data_dir


def main():
    parser = argparse.ArgumentParser(description="Train FrawdLLM")
    parser.add_argument(
        "--config",
        type=str,
        default="tiny",
        choices=["tiny", "small", "base", "tiny-llama", "small-llama"],
        help="Model configuration to use",
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use full TinyStories dataset (default: subset)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    # Get model config
    config = get_config(args.config)
    print(f"Model config: {args.config}")
    print(f"  n_embd: {config.n_embd}")
    print(f"  n_layer: {config.n_layer}")
    print(f"  n_head: {config.n_head}")
    print(f"  context_length: {config.context_length}")

    # Find data
    if args.full_data:
        data_dir = get_data_dir() / "tinystories"
    else:
        data_dir = get_data_dir() / "tinystories_subset"
        if not data_dir.exists():
            data_dir = get_data_dir() / "tinystories"

    train_file = data_dir / "train.txt"
    val_file = data_dir / "validation.txt"

    if not train_file.exists():
        print(f"\nData not found at {data_dir}")
        print("Please run: uv run python -m src.fetch_data.download")
        return

    print(f"\nData directory: {data_dir}")

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        train_file=train_file,
        val_file=val_file,
        context_length=config.context_length,
        batch_size=args.batch_size,
    )

    # Create model
    print("\nCreating model...")
    model = FrawdLLM(config)
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        use_wandb=args.wandb,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}")
        epoch, val_loss = trainer.load_checkpoint(Path(args.resume))
        print(f"Resumed from epoch {epoch}, val_loss={val_loss:.4f}")

    # Train!
    trainer.train()

    # Generate a sample
    print("\n" + "=" * 50)
    print("Generating sample text...")
    print("=" * 50)

    from src.fetch_data.tokenizer import load_tokenizer
    tokenizer = load_tokenizer()

    model.eval()
    prompt = torch.tensor([[config.bos_token_id]], device=trainer.device)
    generated = model.generate(prompt, max_new_tokens=100, temperature=0.8, top_k=50)
    text = tokenizer.decode(generated[0].tolist())
    print(f"\n{text}")


if __name__ == "__main__":
    main()
