"""
SFT training for FrawdLLM 100M on Modal.

Usage:
    # Step 1: Upload SFT data to Modal volume
    modal run scripts/sft_modal_100m.py::upload_data

    # Step 2: Run SFT training
    modal run scripts/sft_modal_100m.py

    # Step 3: Download the SFT checkpoint
    modal run scripts/sft_modal_100m.py::download_checkpoint
"""

import modal

app = modal.App("frawdllm-sft-100m")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "tokenizers",
        "tqdm",
        "wandb",
        "numpy",
    )
)

volume = modal.Volume.from_name("frawdllm-data", create_if_missing=True)
DATA_DIR = "/data"
REPO_URL = "https://github.com/tsingla1998/FrawdLLM.git"


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=600)
def upload_data():
    """Upload local SFT data to Modal volume."""
    from pathlib import Path
    import shutil

    local_sft_dir = Path("data/sft_100m")
    remote_sft_dir = Path(f"{DATA_DIR}/sft_100m")

    if not local_sft_dir.exists():
        print(f"Error: {local_sft_dir} not found locally")
        print("Run: uv run python -m src.fetch_data.generate_sft_100m")
        return {"status": "error", "message": "SFT data not found"}

    # Copy files
    remote_sft_dir.mkdir(parents=True, exist_ok=True)

    for f in local_sft_dir.glob("*.jsonl"):
        print(f"Uploading {f.name}...")
        shutil.copy(f, remote_sft_dir / f.name)

    volume.commit()

    # Count examples
    instructions_file = remote_sft_dir / "instructions.jsonl"
    if instructions_file.exists():
        with open(instructions_file) as f:
            count = sum(1 for _ in f)
        print(f"Uploaded {count:,} SFT examples")
        return {"status": "success", "examples": count}

    return {"status": "success"}


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=300)
def check_data():
    """Check what data is available on Modal volume."""
    from pathlib import Path

    print("Checking Modal volume contents...")

    # Check PT checkpoint
    pt_checkpoint = Path(f"{DATA_DIR}/checkpoints_100m/best.pt")
    print(f"\nPT checkpoint: {'Found' if pt_checkpoint.exists() else 'NOT FOUND'}")
    if pt_checkpoint.exists():
        size_mb = pt_checkpoint.stat().st_size / 1e6
        print(f"  Size: {size_mb:.1f} MB")

    # Check tokenizer
    tokenizer_path = Path(f"{DATA_DIR}/openwebtext/tokenizer.json")
    print(f"\nTokenizer: {'Found' if tokenizer_path.exists() else 'NOT FOUND'}")

    # Check SFT data
    sft_file = Path(f"{DATA_DIR}/sft_100m/instructions.jsonl")
    print(f"\nSFT data: {'Found' if sft_file.exists() else 'NOT FOUND'}")
    if sft_file.exists():
        with open(sft_file) as f:
            count = sum(1 for _ in f)
        print(f"  Examples: {count:,}")

    return {"pt_checkpoint": pt_checkpoint.exists(), "sft_data": sft_file.exists()}


@app.function(
    image=image,
    gpu="A100",
    timeout=6 * 3600,  # 6 hours max
    volumes={DATA_DIR: volume},
)
def train(
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    max_length: int = 512,
    mask_instruction: bool = True,
    gradient_accumulation: int = 4,
):
    """Run SFT training on Modal."""
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

    # Check for required files
    pt_checkpoint = Path(f"{DATA_DIR}/checkpoints_100m/best.pt")
    sft_data = Path(f"{DATA_DIR}/sft_100m/instructions.jsonl")
    tokenizer_path = Path(f"{DATA_DIR}/openwebtext/tokenizer.json")

    if not pt_checkpoint.exists():
        raise RuntimeError(f"PT checkpoint not found: {pt_checkpoint}")
    if not sft_data.exists():
        raise RuntimeError(f"SFT data not found: {sft_data}. Run upload_data first.")
    if not tokenizer_path.exists():
        raise RuntimeError(f"Tokenizer not found: {tokenizer_path}")

    print(f"\n{'='*60}")
    print("SFT Training for FrawdLLM 100M")
    print(f"{'='*60}\n")

    import torch
    import torch.nn as nn

    from src.model.gpt import FrawdLLM
    from src.training.sft_dataset import SFTDataset, add_chat_tokens
    from src.fetch_data.tokenizer import load_tokenizer
    from torch.utils.data import DataLoader

    # Load PT checkpoint
    print(f"Loading PT checkpoint from {pt_checkpoint}...")
    checkpoint = torch.load(pt_checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    print(f"Model config: {config.n_embd}d, {config.n_layer}L, {config.n_head}H")
    print(f"Original vocab size: {config.vocab_size}")

    model = FrawdLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Parameters: {model.count_parameters():,}")

    # Load SFT dataset with custom tokenizer path
    print(f"\nLoading SFT data from {sft_data}...")
    dataset = SFTDataset(
        data_file=sft_data,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        mask_instruction=mask_instruction,
    )

    new_vocab_size = dataset.get_vocab_size()

    # Resize embeddings if needed
    if new_vocab_size != config.vocab_size:
        print(f"\nResizing embeddings: {config.vocab_size} -> {new_vocab_size}")
        old_embeddings = model.embeddings.token_emb
        old_weights = old_embeddings.weight.data

        new_embeddings = nn.Embedding(new_vocab_size, config.n_embd)
        new_embeddings.weight.data[:config.vocab_size] = old_weights
        nn.init.normal_(new_embeddings.weight.data[config.vocab_size:], mean=0.0, std=0.02)

        model.embeddings.token_emb = new_embeddings
        model.lm_head = nn.Linear(config.n_embd, new_vocab_size, bias=False)
        model.lm_head.weight = model.embeddings.token_emb.weight
        config.vocab_size = new_vocab_size

    # Split into train/val
    val_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Split: {train_size:,} train, {val_size:,} validation")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Setup training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * epochs // gradient_accumulation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-7)

    scaler = torch.amp.GradScaler("cuda")

    print(f"\nTraining config:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max length: {max_length}")
    print(f"  Mask instruction: {mask_instruction}")
    print(f"  Total steps: {total_steps}")

    # Training loop
    checkpoint_dir = Path(f"{DATA_DIR}/checkpoints_100m_sft")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0.0
        num_batches = 0

        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            input_ids, targets, loss_mask = batch
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            loss_mask = loss_mask.to(device)

            with torch.amp.autocast("cuda"):
                logits, _ = model(input_ids, None)

                # Masked loss
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                loss_mask_flat = loss_mask.view(-1)

                loss_per_token = torch.nn.functional.cross_entropy(
                    logits_flat, targets_flat, reduction='none'
                )
                masked_loss = loss_per_token * loss_mask_flat
                loss = masked_loss.sum() / (loss_mask_flat.sum() + 1e-8)
                loss = loss / gradient_accumulation

            scaler.scale(loss).backward()

            if (batch_idx + 1) % gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation
            num_batches += 1
            pbar.set_postfix({"loss": f"{total_loss/num_batches:.4f}"})

        train_loss = total_loss / num_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids, targets, loss_mask = batch
                input_ids = input_ids.to(device)
                targets = targets.to(device)
                loss_mask = loss_mask.to(device)

                with torch.amp.autocast("cuda"):
                    logits, _ = model(input_ids, None)
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = targets.view(-1)
                    loss_mask_flat = loss_mask.view(-1)

                    loss_per_token = torch.nn.functional.cross_entropy(
                        logits_flat, targets_flat, reduction='none'
                    )
                    masked_loss = loss_per_token * loss_mask_flat
                    loss = masked_loss.sum() / (loss_mask_flat.sum() + 1e-8)

                val_loss += loss.item()
                val_batches += 1

        val_loss = val_loss / val_batches

        print(f"\nEpoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"New best! Saving checkpoint...")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
                "train_loss": train_loss,
            }, checkpoint_dir / "best.pt")

        # Save epoch checkpoint
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "val_loss": val_loss,
        }, checkpoint_dir / f"epoch_{epoch+1}.pt")

        volume.commit()

    # Generate sample
    print("\n" + "=" * 60)
    print("Testing SFT model:")
    print("=" * 60)

    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer, _, _ = add_chat_tokens(tokenizer)

    model.eval()
    test_prompts = [
        "Explain how photosynthesis works.",
        "What are the benefits of regular exercise?",
        "Write a short poem about the ocean.",
    ]

    for prompt in test_prompts:
        formatted = f"<|bos|><|user|>{prompt}<|assistant|>"
        input_ids = tokenizer.encode(formatted, add_special_tokens=False).ids
        input_len = len(input_ids)
        input_tensor = torch.tensor([input_ids], device=device)

        with torch.no_grad():
            output = model.generate(input_tensor, max_new_tokens=150, temperature=0.8, top_k=50)

        response = tokenizer.decode(output[0][input_len:].tolist())
        response = response.replace("<|eos|>", "").replace("<|pad|>", "").strip()

        print(f"\nQ: {prompt}")
        print(f"A: {response[:300]}...")

    volume.commit()

    return {
        "status": "done",
        "best_val_loss": best_val_loss,
        "epochs": epochs,
        "checkpoint": str(checkpoint_dir / "best.pt"),
    }


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=600)
def download_checkpoint(local_dir: str = "checkpoints_100m_sft"):
    """Download SFT checkpoint from Modal to local."""
    from pathlib import Path
    import shutil

    remote_dir = Path(f"{DATA_DIR}/checkpoints_100m_sft")
    local_path = Path(local_dir)

    if not remote_dir.exists():
        print(f"No SFT checkpoints found at {remote_dir}")
        return {"status": "error"}

    local_path.mkdir(parents=True, exist_ok=True)

    for f in remote_dir.glob("*.pt"):
        print(f"Downloading {f.name}...")
        shutil.copy(f, local_path / f.name)

    print(f"Downloaded to {local_path}")
    return {"status": "success", "path": str(local_path)}


@app.local_entrypoint()
def main(
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-5,
    max_length: int = 512,
):
    """Run SFT training."""
    # First check data
    print("Checking data availability...")
    status = check_data.remote()

    if not status["pt_checkpoint"]:
        print("Error: PT checkpoint not found on Modal volume")
        print("Make sure you've trained the 100M model first")
        return

    if not status["sft_data"]:
        print("Error: SFT data not found on Modal volume")
        print("Run: modal run scripts/sft_modal_100m.py::upload_data")
        return

    print("\nStarting SFT training...")
    result = train.remote(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        max_length=max_length,
    )

    print(f"\nTraining complete!")
    print(f"Best validation loss: {result['best_val_loss']:.4f}")
    print(f"\nTo download checkpoint:")
    print(f"  modal run scripts/sft_modal_100m.py::download_checkpoint")
