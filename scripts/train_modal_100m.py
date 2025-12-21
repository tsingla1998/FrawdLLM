"""
Train FrawdLLM 100M on Modal with OpenWebText.

Usage:
    # Step 1: Prepare data (CPU-only, ~$2)
    modal run scripts/train_modal_100m.py::prepare_data

    # Step 2: Train (A100 GPU)
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
        "datasets==2.21.0",  # Pinned - v3.0+ dropped script-based datasets
        "tokenizers",
        "tqdm",
        "wandb",
        "numpy",
    )
)

volume = modal.Volume.from_name("frawdllm-data", create_if_missing=True)
DATA_DIR = "/data"
REPO_URL = "https://github.com/tsingla1998/FrawdLLM.git"


def prepare_openwebtext_data(output_dir, sample_fraction=0.25, vocab_size=32000):
    """Download and prepare OpenWebText data."""
    from datasets import load_dataset
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from tokenizers.processors import TemplateProcessing
    import numpy as np
    from tqdm import tqdm
    import json

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Preparing OpenWebText ({sample_fraction*100:.0f}% sample)")
    print("=" * 60)

    # Load dataset
    print("\nLoading OpenWebText from HuggingFace...")
    dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)
    total_docs = len(dataset)
    print(f"Total documents: {total_docs:,}")

    # Sample
    sample_size = int(total_docs * sample_fraction)
    print(f"\nSampling {sample_fraction*100:.0f}% = {sample_size:,} documents...")

    dataset = dataset.shuffle(seed=42)
    sampled = dataset.select(range(sample_size))

    # Split train/val
    val_fraction = 0.01
    val_size = int(sample_size * val_fraction)
    train_size = sample_size - val_size

    train_data = sampled.select(range(train_size))
    val_data = sampled.select(range(train_size, sample_size))

    print(f"  Train: {train_size:,} documents")
    print(f"  Val: {val_size:,} documents")

    # Train tokenizer
    print("\n" + "-" * 40)
    tokenizer_sample_size = min(100000, train_size)
    print(f"Training tokenizer on {tokenizer_sample_size:,} texts...")

    tokenizer_texts = [train_data[i]["text"] for i in tqdm(range(tokenizer_sample_size))]

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens,
        show_progress=True,
    )

    tokenizer.train_from_iterator(tokenizer_texts, trainer=trainer)

    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        special_tokens=[
            ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
            ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
        ],
    )
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.save(str(output_dir / "tokenizer.json"))
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # Tokenize helper - uses batch encoding (parallelized in Rust)
    def tokenize_split(data, path, desc):
        bos_id = tokenizer.token_to_id("<|bos|>")
        eos_id = tokenizer.token_to_id("<|eos|>")
        all_tokens = []

        # Process in batches for parallel tokenization
        batch_size = 10000
        texts = []
        for i, example in enumerate(tqdm(data, desc=f"{desc} (collecting)")):
            text = example["text"]
            if text.strip():
                texts.append(text)

            # Encode batch
            if len(texts) >= batch_size:
                encoded_batch = tokenizer.encode_batch(texts)
                for enc in encoded_batch:
                    all_tokens.extend([bos_id] + enc.ids + [eos_id])
                texts = []

        # Final batch
        if texts:
            encoded_batch = tokenizer.encode_batch(texts)
            for enc in encoded_batch:
                all_tokens.extend([bos_id] + enc.ids + [eos_id])

        print(f"  {desc} tokens: {len(all_tokens):,}")
        np.array(all_tokens, dtype=np.uint16).tofile(path)
        return len(all_tokens)

    # Tokenize
    print("\n" + "-" * 40)
    train_tokens = tokenize_split(train_data, output_dir / "train.bin", "Train")
    val_tokens = tokenize_split(val_data, output_dir / "val.bin", "Val")

    # Save metadata
    meta = {
        "vocab_size": vocab_size,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_docs": train_size,
        "val_docs": val_size,
        "sample_fraction": sample_fraction,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone! {train_tokens/1e9:.2f}B train tokens, {val_tokens/1e6:.1f}M val tokens")


@app.function(
    image=image,
    timeout=4 * 3600,  # 4 hours for data prep
    volumes={DATA_DIR: volume},
    cpu=8,  # More CPU cores for tokenization
)
def prepare_data():
    """Prepare OpenWebText data (CPU-only, no GPU cost)."""
    from pathlib import Path

    data_dir = Path(f"{DATA_DIR}/openwebtext")
    if (data_dir / "train.bin").exists():
        print("Data already prepared!")
        return {"status": "already_exists"}

    prepare_openwebtext_data(data_dir)
    volume.commit()
    return {"status": "prepared"}


@app.function(
    image=image,
    gpu="A100",
    timeout=24 * 3600,  # 24 hours max
    volumes={DATA_DIR: volume},
)
def train(
    epochs: int = 1,
    batch_size: int = 8,  # Small batches to fit 1024 context in memory
    learning_rate: float = 3e-4,
    gradient_accumulation: int = 16,  # Effective batch = 8 * 16 = 128
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
        raise RuntimeError("Data not prepared! Run: modal run scripts/train_modal_100m.py::prepare_data")

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


@app.function(image=image, timeout=24 * 3600)  # 30 hours for full pipeline
def run_pipeline(epochs: int = 1, batch_size: int = 32):
    """Run data prep then training (can be detached)."""
    print("Step 1: Preparing data...")
    prep_result = prepare_data.remote()
    print(f"Data prep: {prep_result}")

    print("\nStep 2: Training...")
    result = train.remote(epochs=epochs, batch_size=batch_size)
    print(f"Training complete! Result: {result}")
    return result


@app.local_entrypoint()
def main(
    epochs: int = 1,
    batch_size: int = 32,
):
    """Run full pipeline on Modal."""
    result = run_pipeline.remote(epochs=epochs, batch_size=batch_size)
    print(f"\nDone! Result: {result}")
