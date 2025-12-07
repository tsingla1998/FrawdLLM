"""
Tokenizer training and utilities for FrawdLLM.

This module implements BPE (Byte-Pair Encoding) tokenization from scratch
concepts, using the `tokenizers` library for efficient implementation.

Learning Notes:
--------------
Tokenization is the first step in language modeling. It converts raw text
into a sequence of integers (token IDs) that the model can process.

Key concepts:
1. **Character-level**: Each character is a token. Simple but inefficient.
   - "hello" -> [104, 101, 108, 108, 111] (5 tokens)

2. **Word-level**: Each word is a token. Vocabulary explodes.
   - "hello" -> [12345] (1 token, but vocab size is huge)

3. **Subword (BPE)**: Best of both worlds. Common words are single tokens,
   rare words are split into pieces.
   - "unhappiness" -> ["un", "happiness"] or ["un", "happ", "iness"]

BPE Algorithm (simplified):
1. Start with character-level vocabulary
2. Count all adjacent pairs in the corpus
3. Merge the most frequent pair into a new token
4. Repeat until vocabulary reaches desired size

Why BPE works well:
- Handles rare/new words by breaking into known pieces
- Common words get their own tokens (efficient)
- Fixed vocabulary size (unlike word-level)
"""

import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing


def get_tokenizer_dir() -> Path:
    """Get the tokenizer directory."""
    tokenizer_dir = Path(__file__).parent.parent.parent / "tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    return tokenizer_dir


def text_iterator(file_paths: list[Path], chunk_size: int = 1000) -> Iterator[str]:
    """
    Iterate over text files, yielding chunks for tokenizer training.

    Args:
        file_paths: List of text files to read
        chunk_size: Number of lines to yield at once

    Yields:
        Batches of text lines
    """
    for path in file_paths:
        with open(path, encoding="utf-8") as f:
            batch = []
            for line in f:
                batch.append(line.strip())
                if len(batch) >= chunk_size:
                    yield batch
                    batch = []
            if batch:
                yield batch


def train_bpe_tokenizer(
    train_files: list[Path],
    vocab_size: int = 8192,
    min_frequency: int = 2,
    output_dir: Path | None = None,
) -> Tokenizer:
    """
    Train a BPE tokenizer from scratch.

    Args:
        train_files: List of text files to train on
        vocab_size: Target vocabulary size (smaller = faster, larger = better)
        min_frequency: Minimum frequency for a token to be included
        output_dir: Where to save the tokenizer

    Returns:
        Trained tokenizer

    Learning Notes:
    --------------
    Vocabulary size tradeoffs:
    - Small vocab (4K-8K): Faster training, more tokens per text, may struggle
      with rare words
    - Medium vocab (16K-32K): Good balance for most tasks
    - Large vocab (50K+): Better for diverse text, slower, more memory

    For TinyStories with simple vocabulary, 8K is plenty.
    GPT-2 uses 50K, Llama uses 32K.
    """
    if output_dir is None:
        output_dir = get_tokenizer_dir()

    print(f"Training BPE tokenizer with vocab_size={vocab_size}")
    print(f"Training on: {[str(f) for f in train_files]}")

    # Initialize a BPE tokenizer
    # BPE starts with characters and merges frequent pairs
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenization: Split on whitespace and punctuation first
    # This ensures we don't merge across word boundaries inappropriately
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Special tokens that have specific meanings
    special_tokens = [
        "<|pad|>",      # Padding token (for batching)
        "<|unk|>",      # Unknown token (never used with BPE but good practice)
        "<|bos|>",      # Beginning of sequence
        "<|eos|>",      # End of sequence
        "<|endofstory|>",  # Story boundary (specific to TinyStories)
    ]

    # Trainer handles the BPE algorithm
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    # Train on our text files
    print("\nTraining tokenizer (this may take a few minutes)...")
    tokenizer.train_from_iterator(
        text_iterator(train_files),
        trainer=trainer,
    )

    # Post-processing: Add special tokens to sequences
    # This automatically adds BOS/EOS tokens
    tokenizer.post_processor = TemplateProcessing(
        single="<|bos|> $A <|eos|>",
        special_tokens=[
            ("<|bos|>", tokenizer.token_to_id("<|bos|>")),
            ("<|eos|>", tokenizer.token_to_id("<|eos|>")),
        ],
    )

    # Decoder: Convert tokens back to text
    tokenizer.decoder = decoders.ByteLevel()

    # Save the tokenizer
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    # Also save a human-readable vocab for inspection
    vocab = tokenizer.get_vocab()
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])

    with open(output_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for token, idx in vocab_sorted:
            # Escape special characters for readability
            display_token = repr(token)[1:-1]  # Remove quotes from repr
            f.write(f"{idx}\t{display_token}\n")

    print(f"\nTokenizer saved to {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # Print some examples
    print("\n" + "=" * 60)
    print("Tokenization examples:")
    print("=" * 60)

    examples = [
        "Once upon a time",
        "The quick brown fox jumps over the lazy dog.",
        "Hello, world!",
        "unhappiness",
        "TinyStories is a dataset for training small language models.",
    ]

    for text in examples:
        encoded = tokenizer.encode(text)
        tokens = encoded.tokens
        ids = encoded.ids
        print(f"\nText: {text}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print(f"Length: {len(ids)} tokens")

    return tokenizer


def load_tokenizer(tokenizer_path: Path | None = None) -> Tokenizer:
    """
    Load a trained tokenizer.

    Args:
        tokenizer_path: Path to tokenizer.json. Defaults to tokenizer/tokenizer.json

    Returns:
        Loaded tokenizer
    """
    if tokenizer_path is None:
        tokenizer_path = get_tokenizer_dir() / "tokenizer.json"

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Run `python -m src.fetch_data.tokenizer` to train one."
        )

    return Tokenizer.from_file(str(tokenizer_path))


def analyze_tokenizer(tokenizer: Tokenizer, text_file: Path, num_samples: int = 1000):
    """
    Analyze tokenizer performance on a dataset.

    This helps understand:
    - Average tokens per text (compression ratio)
    - Token frequency distribution
    - Coverage of vocabulary

    Args:
        tokenizer: Trained tokenizer
        text_file: File to analyze
        num_samples: Number of samples to analyze
    """
    print(f"\nAnalyzing tokenizer on {text_file}...")

    token_counts = {}
    total_chars = 0
    total_tokens = 0

    with open(text_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break

            line = line.strip()
            if not line:
                continue

            total_chars += len(line)
            encoded = tokenizer.encode(line)
            total_tokens += len(encoded.ids)

            for token_id in encoded.ids:
                token_counts[token_id] = token_counts.get(token_id, 0) + 1

    # Calculate statistics
    avg_chars_per_token = total_chars / total_tokens if total_tokens > 0 else 0
    unique_tokens_used = len(token_counts)
    vocab_coverage = unique_tokens_used / tokenizer.get_vocab_size() * 100

    print(f"\nTokenizer Analysis ({num_samples} samples):")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Total tokens: {total_tokens:,}")
    print(f"  Avg chars per token: {avg_chars_per_token:.2f}")
    print(f"  Unique tokens used: {unique_tokens_used:,}")
    print(f"  Vocabulary coverage: {vocab_coverage:.1f}%")

    # Most common tokens
    print("\n  Top 20 most common tokens:")
    sorted_tokens = sorted(token_counts.items(), key=lambda x: -x[1])[:20]
    for token_id, count in sorted_tokens:
        token = tokenizer.id_to_token(token_id)
        print(f"    {token!r}: {count}")


if __name__ == "__main__":
    import argparse
    from .download import get_data_dir

    parser = argparse.ArgumentParser(description="Train BPE tokenizer for FrawdLLM")
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=8192,
        help="Vocabulary size (default: 8192)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing train.txt",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze tokenizer after training",
    )

    args = parser.parse_args()

    # Find training data
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Try subset first, then full dataset
        data_dir = get_data_dir() / "tinystories_subset"
        if not data_dir.exists():
            data_dir = get_data_dir() / "tinystories"

    # Use all_data.txt for tokenizer training (no train/val split needed)
    # Tokenizer benefits from seeing ALL data for better vocabulary coverage
    all_data_file = data_dir / "all_data.txt"

    if not all_data_file.exists():
        print(f"Training data not found at {all_data_file}")
        print("Run `python -m src.fetch_data.download` first to download data.")
        exit(1)

    # Train tokenizer on ALL data
    tokenizer = train_bpe_tokenizer(
        train_files=[all_data_file],
        vocab_size=args.vocab_size,
    )

    # Optionally analyze
    if args.analyze:
        analyze_tokenizer(tokenizer, all_data_file)
