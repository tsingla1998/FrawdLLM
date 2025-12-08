"""
DPO (Direct Preference Optimization) training script.

Takes an SFT model and trains it on preference data.

The DPO loss:
    loss = -log(sigmoid(β * (log_π(chosen) - log_π(rejected) - log_ref(chosen) + log_ref(rejected))))

Where:
    π = policy model (being trained)
    ref = reference model (frozen SFT model)
    β = temperature parameter

Usage:
    uv run python scripts/dpo.py --checkpoint checkpoints/sft/best.pt
"""

import argparse
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm

from src.model.gpt import FrawdLLM
from src.training.dpo_dataset import create_dpo_dataloaders, USER_TOKEN, ASSISTANT_TOKEN
from src.fetch_data.tokenizer import load_tokenizer


def load_model(checkpoint_path: Path, device: str = "cpu") -> FrawdLLM:
    """Load model from checkpoint."""
    print(f"Loading {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint["config"]
    model = FrawdLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"  Config: {config.n_embd}d, {config.n_layer}L, vocab={config.vocab_size}")
    return model


def compute_log_probs(
    model: FrawdLLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute sum of log probabilities for each sequence.

    Args:
        model: The language model
        input_ids: [batch, seq_len] input token IDs
        labels: [batch, seq_len] target token IDs (-100 = ignore)

    Returns:
        [batch] sum of log probs for non-ignored tokens
    """
    logits, _ = model(input_ids, None)

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    # Compute per-token log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Gather log probs for actual tokens
    # [batch, seq_len-1]
    token_log_probs = torch.gather(
        log_probs,
        dim=-1,
        index=shift_labels.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    # Mask out ignored positions
    mask = (shift_labels != -100).float()
    token_log_probs = token_log_probs * mask

    # Sum log probs per sequence
    return token_log_probs.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Compute DPO loss.

    Args:
        policy_chosen_logps: [batch] log probs of chosen under policy
        policy_rejected_logps: [batch] log probs of rejected under policy
        ref_chosen_logps: [batch] log probs of chosen under reference
        ref_rejected_logps: [batch] log probs of rejected under reference
        beta: Temperature parameter

    Returns:
        loss: scalar loss
        metrics: dict of useful metrics
    """
    # Log ratios
    policy_ratio = policy_chosen_logps - policy_rejected_logps
    ref_ratio = ref_chosen_logps - ref_rejected_logps

    # DPO loss
    logits = beta * (policy_ratio - ref_ratio)
    loss = -F.logsigmoid(logits).mean()

    # Metrics
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        accuracy = (logits > 0).float().mean()

    metrics = {
        "loss": loss.item(),
        "reward_margin": reward_margin.item(),
        "accuracy": accuracy.item(),
        "chosen_reward": chosen_rewards.mean().item(),
        "rejected_reward": rejected_rewards.mean().item(),
    }

    return loss, metrics


class DPOTrainer:
    """Trainer for DPO."""

    def __init__(
        self,
        policy: FrawdLLM,
        reference: FrawdLLM,
        train_loader,
        val_loader,
        beta: float = 0.1,
        learning_rate: float = 1e-6,
        max_epochs: int = 3,
        checkpoint_dir: Path = Path("checkpoints/dpo"),
        device: str = "auto",
    ):
        # Auto-select device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.policy = policy.to(self.device)
        self.reference = reference.to(self.device)

        # Freeze reference model
        for param in self.reference.parameters():
            param.requires_grad = False
        self.reference.eval()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.beta = beta
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        # Scheduler
        total_steps = len(train_loader) * max_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
        )

        # Mixed precision
        self.use_amp = self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking
        self.best_val_loss = float("inf")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.policy.train()

        total_metrics = {"loss": 0, "accuracy": 0, "reward_margin": 0}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")

        for batch in pbar:
            chosen_ids = batch["chosen_input_ids"].to(self.device)
            chosen_labels = batch["chosen_labels"].to(self.device)
            rejected_ids = batch["rejected_input_ids"].to(self.device)
            rejected_labels = batch["rejected_labels"].to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                # Policy log probs
                policy_chosen_logps = compute_log_probs(self.policy, chosen_ids, chosen_labels)
                policy_rejected_logps = compute_log_probs(self.policy, rejected_ids, rejected_labels)

                # Reference log probs (no grad)
                with torch.no_grad():
                    ref_chosen_logps = compute_log_probs(self.reference, chosen_ids, chosen_labels)
                    ref_rejected_logps = compute_log_probs(self.reference, rejected_ids, rejected_labels)

                # DPO loss
                loss, metrics = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=self.beta,
                )

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # Track metrics
            for k, v in metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "acc": f"{metrics['accuracy']:.2%}",
                "margin": f"{metrics['reward_margin']:.3f}",
            })

        # Average
        return {k: v / num_batches for k, v in total_metrics.items()}

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation."""
        self.policy.eval()

        total_metrics = {"loss": 0, "accuracy": 0, "reward_margin": 0}
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating"):
            chosen_ids = batch["chosen_input_ids"].to(self.device)
            chosen_labels = batch["chosen_labels"].to(self.device)
            rejected_ids = batch["rejected_input_ids"].to(self.device)
            rejected_labels = batch["rejected_labels"].to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                policy_chosen_logps = compute_log_probs(self.policy, chosen_ids, chosen_labels)
                policy_rejected_logps = compute_log_probs(self.policy, rejected_ids, rejected_labels)
                ref_chosen_logps = compute_log_probs(self.reference, chosen_ids, chosen_labels)
                ref_rejected_logps = compute_log_probs(self.reference, rejected_ids, rejected_labels)

                _, metrics = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=self.beta,
                )

            for k, v in metrics.items():
                if k in total_metrics:
                    total_metrics[k] += v
            num_batches += 1

        return {k: v / num_batches for k, v in total_metrics.items()}

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.policy.state_dict(),
            "config": self.policy.config,
            "val_loss": val_loss,
        }

        # Save epoch checkpoint
        path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    def train(self):
        """Full training loop."""
        print(f"\nStarting DPO training")
        print(f"  Beta: {self.beta}")
        print(f"  Epochs: {self.max_epochs}")
        print(f"  Train batches: {len(self.train_loader)}")
        print(f"  Val batches: {len(self.val_loader)}")

        for epoch in range(self.max_epochs):
            print(f"\nEpoch {epoch+1}/{self.max_epochs}")

            # Train
            train_metrics = self.train_epoch(epoch)
            print(f"  Train - loss: {train_metrics['loss']:.4f}, acc: {train_metrics['accuracy']:.2%}")

            # Validate
            val_metrics = self.validate()
            print(f"  Val   - loss: {val_metrics['loss']:.4f}, acc: {val_metrics['accuracy']:.2%}")

            # Checkpoint
            is_best = val_metrics["loss"] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics["loss"]
            self.save_checkpoint(epoch, val_metrics["loss"], is_best)

        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="DPO training for FrawdLLM")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to SFT model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/dpo/preferences.jsonl",
        help="Path to DPO data file",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (smaller than SFT due to memory)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-6,
        help="Learning rate (very low for DPO)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO temperature parameter",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/dpo",
        help="Output directory",
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
        print(f"DPO data not found: {data_path}")
        print("Run: uv run python -m src.fetch_data.generate_dpo")
        return

    # Load policy model
    policy = load_model(checkpoint_path)

    # Create reference model (deep copy, frozen)
    print("Creating reference model (frozen copy)...")
    reference = load_model(checkpoint_path)

    # Load data
    print(f"\nLoading DPO data from {data_path}")
    train_loader, val_loader, vocab_size = create_dpo_dataloaders(
        data_path,
        max_length=512,
        batch_size=args.batch_size,
    )

    # Train
    trainer = DPOTrainer(
        policy=policy,
        reference=reference,
        train_loader=train_loader,
        val_loader=val_loader,
        beta=args.beta,
        learning_rate=args.lr,
        max_epochs=args.epochs,
        checkpoint_dir=output_dir,
    )

    trainer.train()

    # Test generation
    print("\n" + "=" * 50)
    print("Testing DPO model...")
    print("=" * 50)

    tokenizer = load_tokenizer()
    tokenizer.add_special_tokens([USER_TOKEN, ASSISTANT_TOKEN])

    policy.eval()
    prompt = "Write a story about a brave little mouse"
    formatted = f"<|bos|>{USER_TOKEN}{prompt}{ASSISTANT_TOKEN}"

    input_ids = tokenizer.encode(formatted, add_special_tokens=False).ids
    input_tensor = torch.tensor([input_ids], device=trainer.device)

    with torch.no_grad():
        output = policy.generate(input_tensor, max_new_tokens=150, temperature=0.8)

    text = tokenizer.decode(output[0].tolist())
    print(f"\nPrompt: {prompt}")
    print(f"\nGenerated:\n{text}")


if __name__ == "__main__":
    main()
