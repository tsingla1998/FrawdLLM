"""
Download and prepare OpenWebText for pre-training.

OpenWebText is a recreation of OpenAI's WebText dataset - web pages linked from Reddit
with at least 3 karma. ~8M documents, ~8B tokens.

We'll use 25% (~2B tokens) for compute-optimal training of 100M params.

Usage:
    uv run python -m src.fetch_data.prepare_openwebtext

This will:
1. Train a new 32K BPE tokenizer on a sample
2. Download and tokenize 25% of OpenWebText
3. Save to data/openwebtext/
"""

from pathlib import Path
from datasets import load_dataset
from tokenizers import Tokenizer
import numpy as np
from tqdm import tqdm
import json

from .tokenizer import train_bpe_from_texts

# Output paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "openwebtext"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
TRAIN_PATH = DATA_DIR / "train.bin"
VAL_PATH = DATA_DIR / "val.bin"
META_PATH = DATA_DIR / "meta.json"

# Config
VOCAB_SIZE = 32000
SAMPLE_FRACTION = 0.25  # Use 25% of OpenWebText
VAL_FRACTION = 0.01     # 1% for validation


def tokenize_and_save(
    dataset,
    tokenizer: Tokenizer,
    output_path: Path,
    desc: str = "Tokenizing",
) -> int:
    """Tokenize dataset and save as memory-mapped numpy array."""

    # First pass: count total tokens
    print(f"Counting tokens for {desc}...")
    total_tokens = 0
    all_tokens = []

    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")

    for example in tqdm(dataset, desc=desc):
        text = example["text"]
        if not text.strip():
            continue

        # Tokenize
        encoded = tokenizer.encode(text)
        tokens = [bos_id] + encoded.ids + [eos_id]
        all_tokens.extend(tokens)

    total_tokens = len(all_tokens)
    print(f"  Total tokens: {total_tokens:,}")

    # Save as numpy memmap
    print(f"Saving to {output_path}...")
    arr = np.array(all_tokens, dtype=np.uint16)  # uint16 supports up to 65535 vocab
    arr.tofile(output_path)

    return total_tokens


def main():
    """Main entry point."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Preparing OpenWebText (25% sample)")
    print("=" * 60)

    # Load dataset
    print("\nLoading OpenWebText from HuggingFace...")
    print("(This may take a while on first run - ~8GB download)")

    dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)
    total_docs = len(dataset)
    print(f"Total documents: {total_docs:,}")

    # Sample 25%
    sample_size = int(total_docs * SAMPLE_FRACTION)
    print(f"\nSampling {SAMPLE_FRACTION*100:.0f}% = {sample_size:,} documents...")

    # Shuffle and sample
    dataset = dataset.shuffle(seed=42)
    sampled = dataset.select(range(sample_size))

    # Split into train/val
    val_size = int(sample_size * VAL_FRACTION)
    train_size = sample_size - val_size

    train_data = sampled.select(range(train_size))
    val_data = sampled.select(range(train_size, sample_size))

    print(f"  Train: {train_size:,} documents")
    print(f"  Val: {val_size:,} documents")

    # Train tokenizer on a sample of training data
    print("\n" + "-" * 40)
    tokenizer_sample_size = min(100000, train_size)
    print(f"Collecting {tokenizer_sample_size:,} texts for tokenizer training...")
    tokenizer_texts = [train_data[i]["text"] for i in tqdm(range(tokenizer_sample_size))]

    tokenizer = train_bpe_from_texts(
        tokenizer_texts,
        vocab_size=VOCAB_SIZE,
        output_path=TOKENIZER_PATH,
    )

    # Tokenize train set
    print("\n" + "-" * 40)
    train_tokens = tokenize_and_save(train_data, tokenizer, TRAIN_PATH, "Train")

    # Tokenize val set
    print("\n" + "-" * 40)
    val_tokens = tokenize_and_save(val_data, tokenizer, VAL_PATH, "Val")

    # Save metadata
    meta = {
        "vocab_size": VOCAB_SIZE,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "train_docs": train_size,
        "val_docs": val_size,
        "sample_fraction": SAMPLE_FRACTION,
        "tokenizer_path": str(TOKENIZER_PATH),
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 60)
    print("Done!")
    print(f"  Train tokens: {train_tokens:,} ({train_tokens/1e9:.2f}B)")
    print(f"  Val tokens: {val_tokens:,} ({val_tokens/1e6:.1f}M)")
    print(f"  Output dir: {DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
