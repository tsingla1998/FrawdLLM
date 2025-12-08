"""
SFT (Supervised Fine-Tuning) script for FrawdLLM.

Takes a pre-trained model and fine-tunes it on instruction-response pairs.

Usage:
    uv run python scripts/sft.py --checkpoint checkpoints/best.pt
    uv run python scripts/sft.py --checkpoint checkpoints/best.pt --mask-instruction
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.model.gpt import FrawdLLM
from src.training.sft_dataset import create_sft_dataloaders, USER_TOKEN, ASSISTANT_TOKEN
from src.training.trainer import Trainer
from src.fetch_data.tokenizer import load_tokenizer


def resize_embeddings(model: FrawdLLM, new_vocab_size: int) -> FrawdLLM:
    """
    Resize model embeddings for new vocabulary size.

    This is needed when we add new tokens (like <|user|>, <|assistant|>).

    Args:
        model: The pre-trained model
        new_vocab_size: New vocabulary size (original + new tokens)

    Returns:
        Model with resized embeddings
    """
    old_vocab_size = model.config.vocab_size

    if new_vocab_size == old_vocab_size:
        return model

    print(f"Resizing embeddings: {old_vocab_size} -> {new_vocab_size}")

    # Get old embeddings
    old_embeddings = model.embeddings.token_emb
    old_weights = old_embeddings.weight.data

    # Create new embedding layer
    new_embeddings = nn.Embedding(new_vocab_size, model.config.n_embd)

    # Copy old weights
    new_embeddings.weight.data[:old_vocab_size] = old_weights

    # Initialize new tokens with small random values
    # (similar to how the original embeddings were initialized)
    nn.init.normal_(new_embeddings.weight.data[old_vocab_size:], mean=0.0, std=0.02)

    # Replace embedding layer
    model.embeddings.token_emb = new_embeddings

    # Also update the output head (shares weights with embeddings via weight tying)
    # The lm_head should point to the same weights
    model.lm_head = nn.Linear(model.config.n_embd, new_vocab_size, bias=False)
    model.lm_head.weight = model.embeddings.token_emb.weight

    # Update config
    model.config.vocab_size = new_vocab_size

    print(f"  Added {new_vocab_size - old_vocab_size} new token embeddings")

    return model


def load_pretrained_model(checkpoint_path: Path, device: str = "cpu") -> FrawdLLM:
    """
    Load a pre-trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load onto

    Returns:
        Loaded model
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    print(f"Model config: {config.n_embd}d, {config.n_layer}L, {config.n_head}H")

    model = FrawdLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded model with {model.count_parameters():,} parameters")

    return model


class SFTTrainer(Trainer):
    """
    Trainer modified for SFT with loss masking support.
    """

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch with optional loss masking."""
        from tqdm import tqdm
        from torch.amp import autocast

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")

        for batch in pbar:
            # Unpack batch - may have loss_mask
            if len(batch) == 3:
                input_ids, targets, loss_mask = batch
                loss_mask = loss_mask.to(self.device)
            else:
                input_ids, targets = batch
                loss_mask = None

            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits, _ = self.model(input_ids, None)  # Don't use built-in loss

                # Compute loss with optional masking
                if loss_mask is not None:
                    # Flatten for cross-entropy
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = targets.view(-1)
                    loss_mask_flat = loss_mask.view(-1)

                    # Compute per-token loss
                    loss_per_token = torch.nn.functional.cross_entropy(
                        logits_flat, targets_flat, reduction='none'
                    )

                    # Apply mask and average
                    masked_loss = loss_per_token * loss_mask_flat
                    loss = masked_loss.sum() / (loss_mask_flat.sum() + 1e-8)
                else:
                    # Standard loss
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=self.config.pad_token_id,
                    )

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Update weights
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Track loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            avg_loss = total_loss / num_batches
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
            })

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        """Run validation with optional loss masking."""
        from tqdm import tqdm
        from torch.amp import autocast

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            # Unpack batch
            if len(batch) == 3:
                input_ids, targets, loss_mask = batch
                loss_mask = loss_mask.to(self.device)
            else:
                input_ids, targets = batch
                loss_mask = None

            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                logits, _ = self.model(input_ids, None)

                if loss_mask is not None:
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = targets.view(-1)
                    loss_mask_flat = loss_mask.view(-1)

                    loss_per_token = torch.nn.functional.cross_entropy(
                        logits_flat, targets_flat, reduction='none'
                    )
                    masked_loss = loss_per_token * loss_mask_flat
                    loss = masked_loss.sum() / (loss_mask_flat.sum() + 1e-8)
                else:
                    loss = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=self.config.pad_token_id,
                    )

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="SFT training for FrawdLLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to pre-trained model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/sft/instructions.jsonl",
        help="Path to SFT data file",
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
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (lower than PT since we're fine-tuning)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--mask-instruction",
        action="store_true",
        help="Only compute loss on response tokens (not instruction)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/sft",
        help="Output directory for checkpoints",
    )

    args = parser.parse_args()

    # Paths
    checkpoint_path = Path(args.checkpoint)
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    if not data_path.exists():
        print(f"SFT data not found: {data_path}")
        print("Run: uv run python -m src.fetch_data.generate_sft")
        return

    # Load model
    model = load_pretrained_model(checkpoint_path)

    # Create dataloaders (this also adds chat tokens to tokenizer)
    print(f"\nLoading SFT data from {data_path}")
    train_loader, val_loader, new_vocab_size = create_sft_dataloaders(
        train_file=data_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        mask_instruction=args.mask_instruction,
    )

    # Resize embeddings for new tokens
    model = resize_embeddings(model, new_vocab_size)

    print(f"\nSFT Training Config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Max length: {args.max_length}")
    print(f"  Mask instruction: {args.mask_instruction}")
    print(f"  Output dir: {output_dir}")

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=model.config,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        checkpoint_dir=output_dir,
    )

    # Train!
    trainer.train()

    # Generate a sample
    print("\n" + "=" * 50)
    print("Testing SFT model with a prompt...")
    print("=" * 50)

    tokenizer = load_tokenizer()
    tokenizer.add_special_tokens([USER_TOKEN, ASSISTANT_TOKEN])

    model.eval()

    # Create a test prompt
    test_instruction = "Write a short story about a friendly dragon."
    prompt = f"<|bos|><|user|>{test_instruction}<|assistant|>"

    input_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    input_tensor = torch.tensor([input_ids], device=trainer.device)

    generated = model.generate(input_tensor, max_new_tokens=150, temperature=0.8, top_k=50)
    text = tokenizer.decode(generated[0].tolist())

    print(f"\nPrompt: {test_instruction}")
    print(f"\nGenerated:\n{text}")


if __name__ == "__main__":
    main()
