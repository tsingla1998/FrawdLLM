"""
Train FrawdLLM 100M on Modal with OpenWebText.

Usage:
    # Prepare data locally first (recommended)
    uv run python -m src.fetch_data.prepare_openwebtext

    # Upload data to Modal volume
    modal volume put frawdllm-data data/openwebtext /openwebtext

    # Run training
    modal run scripts/train_modal_100m.py
"""

import modal

app = modal.App("frawdllm-100m")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tokenizers",
        "tqdm",
        "wandb",
        "numpy",
    )
)

volume = modal.Volume.from_name("frawdllm-data", create_if_missing=True)
DATA_DIR = "/data"
REPO_URL = "https://github.com/tsingla1998/FrawdLLM.git"


@app.function(
    image=image,
    gpu="A100",
    timeout=8 * 3600,  # 8 hours max for larger model
    volumes={DATA_DIR: volume},
)
def train(
    epochs: int = 1,
    batch_size: int = 32,  # Smaller batches for 100M model
    learning_rate: float = 3e-4,
    gradient_accumulation: int = 4,  # Effective batch = 32 * 4 = 128
):
    """Train 100M model on OpenWebText."""
    import subprocess
    import os
    import sys
    from pathlib import Path

    # Clone/update repo
    repo_dir = f"{DATA_DIR}/FrawdLLM"
    if not os.path.exists(repo_dir):
        print(f"Cloning {REPO_URL}...")
        subprocess.run(["git", "clone", "-b", "100m-scale-up", REPO_URL, repo_dir], check=True)
    else:
        print("Repo exists, pulling latest...")
        subprocess.run(["git", "-C", repo_dir, "fetch", "origin"], check=True)
        subprocess.run(["git", "-C", repo_dir, "checkout", "100m-scale-up"], check=True)
        subprocess.run(["git", "-C", repo_dir, "pull"], check=True)

    os.chdir(repo_dir)
    sys.path.insert(0, repo_dir)

    # Check for OpenWebText data
    data_dir = Path(f"{DATA_DIR}/openwebtext")
    if not (data_dir / "train.bin").exists():
        print("ERROR: OpenWebText data not found!")
        print("Please prepare data locally and upload:")
        print("  1. uv run python -m src.fetch_data.prepare_openwebtext")
        print("  2. modal volume put frawdllm-data data/openwebtext /openwebtext")
        return {"error": "Data not found"}

    print(f"\n{'='*60}")
    print("Training FrawdLLM 100M on OpenWebText")
    print(f"{'='*60}\n")

    import torch
    import json

    from src.model.gpt import FrawdLLM
    from src.model.config import get_config
    from src.training.openwebtext_dataset import create_openwebtext_dataloaders
    from src.training.trainer import Trainer

    # Load metadata
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)

    print(f"Data: {meta['train_tokens']:,} train tokens, {meta['val_tokens']:,} val tokens")

    # Get 100M config
    model_config = get_config("100m")
    print(f"Model: {model_config.n_embd}d, {model_config.n_layer}L, {model_config.n_head}H")
    print(f"RoPE: {model_config.use_rope}, Context: {model_config.context_length}")

    # Create dataloaders
    train_loader, val_loader, _ = create_openwebtext_dataloaders(
        data_dir=data_dir,
        context_length=model_config.context_length,
        batch_size=batch_size,
        num_workers=4,
    )

    # Create model
    model = FrawdLLM(model_config)
    num_params = model.count_parameters()
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # GPU info
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    # Calculate steps
    total_batches = len(train_loader) * epochs
    print(f"\nTraining:")
    print(f"  Batches per epoch: {len(train_loader):,}")
    print(f"  Total batches: {total_batches:,}")
    print(f"  Gradient accumulation: {gradient_accumulation}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=model_config,
        learning_rate=learning_rate,
        max_epochs=epochs,
        checkpoint_dir=Path(DATA_DIR) / "checkpoints_100m",
        gradient_accumulation_steps=gradient_accumulation,
    )

    # Train
    trainer.train()
    volume.commit()

    # Generate sample
    print("\n" + "=" * 60)
    print("Sample generation:")
    print("=" * 60)

    from src.fetch_data.tokenizer import load_tokenizer
    tokenizer = load_tokenizer(data_dir / "tokenizer.json")

    model.eval()
    prompt_text = "The meaning of life is"
    encoded = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt = torch.tensor([encoded.ids], device=trainer.device)

    generated = model.generate(prompt, max_new_tokens=100, temperature=0.8, top_k=50)
    text = tokenizer.decode(generated[0].tolist())
    print(f"Prompt: {prompt_text}")
    print(f"Generated: {text}")

    return {"status": "done", "best_val_loss": trainer.best_val_loss}


@app.local_entrypoint()
def main(
    epochs: int = 1,
    batch_size: int = 32,
):
    """Run training on Modal."""
    result = train.remote(epochs=epochs, batch_size=batch_size)
    print(f"\nTraining complete! Result: {result}")
