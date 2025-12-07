"""
Data download utilities for FrawdLLM.

This module handles downloading open-source datasets for training.
We start with TinyStories - a high-quality dataset designed to train
small language models that can generate coherent stories.

Learning Notes:
--------------
TinyStories was created by Microsoft Research specifically for training
small language models. It contains ~2.2M synthetic short stories generated
by GPT-3.5/4, designed so that even models with <10M parameters can learn
coherent language generation.

Paper: https://arxiv.org/abs/2305.07759

Why TinyStories is great for learning:
1. Small enough to iterate quickly (~2GB)
2. High quality (no web scraping artifacts)
3. Simple vocabulary (children's stories)
4. Stories have clear structure (beginning, middle, end)
"""

import os
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def get_data_dir() -> Path:
    """Get the data directory, creating it if needed."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_tinystories(
    output_dir: Path | None = None,
    num_proc: int = 4,
    val_ratio: float = 0.01,
) -> Path:
    """
    Download the TinyStories dataset from HuggingFace.

    Creates three files:
    - all_data.txt: ALL stories combined (for tokenizer training)
    - train.txt: 99% of stories (for model training)
    - validation.txt: 1% of stories (for model evaluation)

    Args:
        output_dir: Where to save the dataset. Defaults to data/tinystories/
        num_proc: Number of processes for parallel processing
        val_ratio: Fraction of data to use for validation (default 1%)

    Returns:
        Path to the downloaded dataset directory

    Learning Notes:
    --------------
    Why save all_data.txt separately?
    - Tokenizer training doesn't need train/val split (no overfitting risk)
    - Training on ALL data gives tokenizer better vocabulary coverage
    - Model training DOES need the split to detect overfitting

    HuggingFace `datasets` library handles:
    - Downloading from the Hub
    - Caching (won't re-download if cached)
    - Memory-efficient loading (can stream large datasets)
    - Parallel processing
    """
    if output_dir is None:
        output_dir = get_data_dir() / "tinystories"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading TinyStories dataset...")
    print("This may take a few minutes on first run (~2GB download)")
    print()

    # Load from HuggingFace Hub
    # The dataset is stored as parquet files for efficient loading
    dataset = load_dataset(
        "roneneldan/TinyStories",
        num_proc=num_proc,
    )

    # Combine train + validation from HuggingFace into one pool
    # We'll do our own split for consistency
    all_examples = list(dataset["train"]) + list(dataset["validation"])
    total_count = len(all_examples)

    print(f"\nDataset loaded! Total examples: {total_count:,}")

    # Shuffle deterministically for reproducibility
    import random
    random.seed(42)
    random.shuffle(all_examples)

    # Calculate split point
    val_count = int(total_count * val_ratio)
    train_count = total_count - val_count

    print(f"  Will split into: {train_count:,} train / {val_count:,} validation")

    # File paths
    all_data_path = output_dir / "all_data.txt"
    train_path = output_dir / "train.txt"
    val_path = output_dir / "validation.txt"

    # Save ALL data first (for tokenizer training)
    print(f"\nSaving all_data.txt (for tokenizer training)...")
    with open(all_data_path, "w", encoding="utf-8") as f:
        for example in tqdm(all_examples, desc="Writing all_data.txt"):
            text = example["text"].strip()
            f.write(text + "\n<|endofstory|>\n")

    # Save training split
    print("Saving train.txt (for model training)...")
    with open(train_path, "w", encoding="utf-8") as f:
        for example in tqdm(all_examples[:train_count], desc="Writing train.txt"):
            text = example["text"].strip()
            f.write(text + "\n<|endofstory|>\n")

    # Save validation split
    print("Saving validation.txt (for model evaluation)...")
    with open(val_path, "w", encoding="utf-8") as f:
        for example in tqdm(all_examples[train_count:], desc="Writing validation.txt"):
            text = example["text"].strip()
            f.write(text + "\n<|endofstory|>\n")

    # Print some stats
    all_size = all_data_path.stat().st_size / (1024 * 1024 * 1024)
    train_size = train_path.stat().st_size / (1024 * 1024 * 1024)
    val_size = val_path.stat().st_size / (1024 * 1024 * 1024)

    print(f"\nDataset saved!")
    print(f"  {all_data_path}: {all_size:.2f} GB (for tokenizer)")
    print(f"  {train_path}: {train_size:.2f} GB (for model training)")
    print(f"  {val_path}: {val_size:.3f} GB (for model evaluation)")

    # Show a sample story
    print("\n" + "=" * 60)
    print("Sample story from the dataset:")
    print("=" * 60)
    sample = all_examples[0]["text"]
    print(sample[:500] + "..." if len(sample) > 500 else sample)
    print("=" * 60)

    return output_dir


def download_subset(
    num_examples: int = 10000,
    output_dir: Path | None = None,
    val_ratio: float = 0.1,
) -> Path:
    """
    Download a small subset for quick local testing.

    Creates three files:
    - all_data.txt: ALL examples (for tokenizer training)
    - train.txt: 90% of examples (for model training)
    - validation.txt: 10% of examples (for model evaluation)

    This is useful for:
    - Testing the training pipeline quickly
    - Debugging on CPU/M3 without waiting for full dataset
    - Iterating on model architecture

    Args:
        num_examples: Number of total examples to download
        output_dir: Where to save. Defaults to data/tinystories_subset/
        val_ratio: Fraction for validation (default 10% for small datasets)

    Returns:
        Path to the subset directory
    """
    if output_dir is None:
        output_dir = get_data_dir() / "tinystories_subset"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {num_examples:,} example subset...")

    # Download examples
    dataset = load_dataset(
        "roneneldan/TinyStories",
        split=f"train[:{num_examples}]",
    )

    all_examples = list(dataset)

    # Shuffle deterministically
    import random
    random.seed(42)
    random.shuffle(all_examples)

    # Split
    val_count = int(num_examples * val_ratio)
    train_count = num_examples - val_count

    print(f"  Splitting into: {train_count:,} train / {val_count:,} validation")

    # File paths
    all_data_path = output_dir / "all_data.txt"
    train_path = output_dir / "train.txt"
    val_path = output_dir / "validation.txt"

    # Save all data (for tokenizer)
    with open(all_data_path, "w", encoding="utf-8") as f:
        for example in tqdm(all_examples, desc="Writing all_data.txt"):
            f.write(example["text"].strip() + "\n<|endofstory|>\n")

    # Save train split
    with open(train_path, "w", encoding="utf-8") as f:
        for example in tqdm(all_examples[:train_count], desc="Writing train.txt"):
            f.write(example["text"].strip() + "\n<|endofstory|>\n")

    # Save validation split
    with open(val_path, "w", encoding="utf-8") as f:
        for example in tqdm(all_examples[train_count:], desc="Writing validation.txt"):
            f.write(example["text"].strip() + "\n<|endofstory|>\n")

    print(f"\nSubset saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download training data for FrawdLLM")
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Download only N examples (for quick testing)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Download the full TinyStories dataset",
    )

    args = parser.parse_args()

    if args.subset:
        download_subset(num_examples=args.subset)
    elif args.full:
        download_tinystories()
    else:
        # Default: download a 10K subset for quick iteration
        print("Downloading 10K subset for quick testing...")
        print("Use --full for the complete dataset")
        print()
        download_subset(num_examples=10000)
