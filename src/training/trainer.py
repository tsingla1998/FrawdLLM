"""
Training loop for FrawdLLM.

This handles:
1. Forward pass (get predictions)
2. Compute loss (how wrong are we?)
3. Backward pass (compute gradients)
4. Update weights (optimizer step)
5. Logging and checkpointing

Supports mixed precision (fp16) for ~2x speedup on GPUs.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import time
import math

# Optional: Weights & Biases for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.model.gpt import FrawdLLM
from src.model.config import ModelConfig


class Trainer:
    """
    Trainer for FrawdLLM.

    Handles the training loop, validation, checkpointing, and logging.
    """

    def __init__(
        self,
        model: FrawdLLM,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: ModelConfig,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.1,
        max_epochs: int = 10,
        grad_clip: float = 1.0,
        checkpoint_dir: Path | None = None,
        use_wandb: bool = False,
        wandb_project: str = "frawdllm",
        use_amp: bool | None = None,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Args:
            model: The FrawdLLM model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Model configuration
            learning_rate: Peak learning rate
            weight_decay: L2 regularization strength
            max_epochs: Number of training epochs
            grad_clip: Maximum gradient norm (prevents exploding gradients)
            checkpoint_dir: Where to save checkpoints
            use_wandb: Whether to log to Weights & Biases
            wandb_project: W&B project name
            use_amp: Use automatic mixed precision (default: auto-detect CUDA)
            gradient_accumulation_steps: Accumulate gradients over N steps
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Setup device (MPS for Mac, CUDA for Nvidia, else CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using CUDA ({torch.cuda.get_device_name()})")
        else:
            self.device = torch.device("cpu")
            print("Using CPU (training will be slow)")

        self.model.to(self.device)

        # Mixed precision training (fp16) - ~2x speedup on CUDA
        # Auto-enable on CUDA if not specified
        if use_amp is None:
            use_amp = self.device.type == "cuda"
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp)
        if use_amp:
            print("Using mixed precision (fp16)")

        # Optimizer: AdamW with weight decay
        # We separate parameters that should/shouldn't have weight decay
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases and layer norms
                if 'bias' in name or 'ln' in name or 'layernorm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        optimizer_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        self.optimizer = AdamW(optimizer_groups, lr=learning_rate)

        # Learning rate scheduler: Cosine decay
        total_steps = len(train_loader) * max_epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # Checkpointing
        if checkpoint_dir is None:
            checkpoint_dir = Path(__file__).parent.parent.parent / "checkpoints"
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                config={
                    "vocab_size": config.vocab_size,
                    "n_embd": config.n_embd,
                    "n_layer": config.n_layer,
                    "n_head": config.n_head,
                    "context_length": config.context_length,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "max_epochs": max_epochs,
                    "batch_size": train_loader.batch_size,
                },
            )

        # Track best validation loss for checkpointing
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")

        for batch_idx, (input_ids, targets) in enumerate(pbar):
            # Move to device
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            # Forward pass with mixed precision
            with autocast(enabled=self.use_amp):
                logits, loss = self.model(input_ids, targets)
                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass (accumulate gradients)
            self.scaler.scale(loss).backward()

            # Track loss (unscaled for logging)
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Only step optimizer every N batches
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping (prevents exploding gradients)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                # Clear gradients for next accumulation
                self.optimizer.zero_grad()

            # Update progress bar
            avg_loss = total_loss / num_batches
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{current_lr:.2e}',
            })

            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item() * self.gradient_accumulation_steps,
                    'train/lr': current_lr,
                    'train/step': epoch * len(self.train_loader) + batch_idx,
                })

        return total_loss / num_batches

    @torch.no_grad()
    def validate(self) -> float:
        """
        Run validation.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for input_ids, targets in tqdm(self.val_loader, desc="Validating"):
            input_ids = input_ids.to(self.device)
            targets = targets.to(self.device)

            with autocast(enabled=self.use_amp):
                logits, loss = self.model(input_ids, targets)

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save a model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
        }

        # Save latest
        path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, path)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    def load_checkpoint(self, path: Path):
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['val_loss']

    def train(self):
        """
        Full training loop.
        """
        print(f"\nStarting training for {self.max_epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset):,}")
        print(f"Validation samples: {len(self.val_loader.dataset):,}")
        print(f"Batch size: {self.train_loader.batch_size}")
        print(f"Total steps: {len(self.train_loader) * self.max_epochs:,}")
        print()

        start_time = time.time()

        for epoch in range(self.max_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Check if this is the best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Save checkpoint
            self.save_checkpoint(epoch, val_loss, is_best)

            # Log
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.max_epochs}:")
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Val loss:   {val_loss:.4f}")
            print(f"  Time:       {epoch_time:.1f}s")

            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_loss,
                    'val/loss': val_loss,
                    'epoch_time': epoch_time,
                })

        total_time = time.time() - start_time
        print(f"\nTraining complete!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best val loss: {self.best_val_loss:.4f}")

        if self.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    print("Trainer module loaded successfully!")
    print("Run training with: uv run python scripts/train.py")
