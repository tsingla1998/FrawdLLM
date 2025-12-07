"""
Train FrawdLLM on Modal (cloud GPUs).

Usage:
    # First time: install modal and authenticate
    pip install modal
    modal setup

    # Run training
    modal run scripts/train_modal.py

    # Run with custom config
    modal run scripts/train_modal.py --config small --epochs 3
"""

import modal

# Define the Modal app
app = modal.App("frawdllm")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tokenizers",
        "tqdm",
        "wandb",
    )
)

# Create a volume to persist data and checkpoints across runs
volume = modal.Volume.from_name("frawdllm-data", create_if_missing=True)

DATA_DIR = "/data"
REPO_URL = "https://github.com/tsingla1998/FrawdLLM.git"


@app.function(
    image=image,
    gpu="A10G",  # Options: "T4", "A10G", "A100", "H100"
    timeout=4 * 3600,  # 4 hours max
    volumes={DATA_DIR: volume},
)
def train(
    config: str = "small",
    epochs: int = 2,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
):
    """
    Train FrawdLLM on a cloud GPU.

    Args:
        config: Model config (tiny, small, base)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    import subprocess
    import os
    import sys

    # Clone the repo
    repo_dir = f"{DATA_DIR}/FrawdLLM"
    if not os.path.exists(repo_dir):
        print(f"Cloning {REPO_URL}...")
        subprocess.run(["git", "clone", REPO_URL, repo_dir], check=True)
    else:
        print("Repo already exists, pulling latest...")
        subprocess.run(["git", "-C", repo_dir, "pull"], check=True)

    # Change to repo directory
    os.chdir(repo_dir)
    sys.path.insert(0, repo_dir)

    # Download data if not present
    data_dir = f"{DATA_DIR}/tinystories"
    if not os.path.exists(f"{data_dir}/train.txt"):
        print("Downloading TinyStories dataset...")
        from src.fetch_data.download import download_tinystories
        download_tinystories(output_dir=data_dir)

    # Train tokenizer if not present
    tokenizer_dir = f"{DATA_DIR}/tokenizer"
    if not os.path.exists(f"{tokenizer_dir}/tokenizer.json"):
        print("Training tokenizer...")
        from src.fetch_data.tokenizer import train_bpe_tokenizer
        from pathlib import Path
        train_bpe_tokenizer(
            train_files=[Path(f"{data_dir}/all_data.txt")],
            output_dir=Path(tokenizer_dir),
        )

    # Now run training
    print(f"\n{'='*50}")
    print(f"Starting training: config={config}, epochs={epochs}")
    print(f"{'='*50}\n")

    import torch
    from pathlib import Path

    # Patch paths to use our volume
    os.environ["FRAWDLLM_DATA_DIR"] = DATA_DIR

    from src.model.gpt import FrawdLLM
    from src.model.config import get_config
    from src.training.dataset import create_dataloaders
    from src.training.trainer import Trainer

    # Get config
    model_config = get_config(config)
    print(f"Model: {model_config.n_embd}d, {model_config.n_layer}L, {model_config.n_head}H")

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_file=Path(f"{data_dir}/train.txt"),
        val_file=Path(f"{data_dir}/validation.txt"),
        context_length=model_config.context_length,
        batch_size=batch_size,
    )

    # Create model
    model = FrawdLLM(model_config)
    num_params = model.count_parameters()
    print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=model_config,
        learning_rate=learning_rate,
        max_epochs=epochs,
        checkpoint_dir=Path(f"{DATA_DIR}/checkpoints"),
    )

    # Train
    trainer.train()

    # Save final model to volume
    volume.commit()

    # Generate sample
    print("\n" + "=" * 50)
    print("Sample generation:")
    print("=" * 50)

    from src.fetch_data.tokenizer import load_tokenizer
    tokenizer = load_tokenizer(Path(f"{tokenizer_dir}/tokenizer.json"))

    model.eval()
    prompt = torch.tensor([[model_config.bos_token_id]], device=trainer.device)
    generated = model.generate(prompt, max_new_tokens=100, temperature=0.8, top_k=50)
    text = tokenizer.decode(generated[0].tolist())
    print(text)

    return {"status": "done", "best_val_loss": trainer.best_val_loss}


@app.function(
    image=image,
    gpu="A10G",
    volumes={DATA_DIR: volume},
)
def generate(prompt: str = "Once upon a time", max_tokens: int = 100):
    """Generate text from a trained model."""
    import torch
    import os
    import sys

    repo_dir = f"{DATA_DIR}/FrawdLLM"
    os.chdir(repo_dir)
    sys.path.insert(0, repo_dir)

    from pathlib import Path
    from src.model.gpt import FrawdLLM
    from src.model.config import get_config
    from src.fetch_data.tokenizer import load_tokenizer

    # Load model
    checkpoint_path = Path(f"{DATA_DIR}/checkpoints/best.pt")
    if not checkpoint_path.exists():
        return {"error": "No trained model found. Run training first."}

    checkpoint = torch.load(checkpoint_path)
    config = checkpoint["config"]

    model = FrawdLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model.cuda()

    # Load tokenizer
    tokenizer = load_tokenizer(Path(f"{DATA_DIR}/tokenizer/tokenizer.json"))

    # Encode prompt
    encoded = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoded.ids], device="cuda")

    # Generate
    output_ids = model.generate(input_ids, max_new_tokens=max_tokens, temperature=0.8, top_k=50)
    text = tokenizer.decode(output_ids[0].tolist())

    return {"generated": text}


@app.local_entrypoint()
def main(
    config: str = "small",
    epochs: int = 2,
    batch_size: int = 32,
):
    """Local entrypoint - runs on Modal cloud."""
    result = train.remote(config=config, epochs=epochs, batch_size=batch_size)
    print(f"\nTraining complete! Result: {result}")
