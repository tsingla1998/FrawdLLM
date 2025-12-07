"""
Dataset for FrawdLLM training.

Loads tokenized text and creates training examples.

For language modeling, we don't have separate inputs/outputs.
Instead:
    Input:  [token0, token1, token2, token3]
    Target: [token1, token2, token3, token4]

The model learns to predict the next token at each position.

Optimizations:
- Caches tokenized data to disk (tokenize once, load instantly)
- Uses parallel tokenization for speed
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import hashlib
import os

from src.fetch_data.tokenizer import load_tokenizer


def get_cache_path(text_file: Path) -> Path:
    """
    Get cache file path for tokenized data.

    Uses a hash of the filename to avoid collisions.
    Cache stored in same directory as text file.
    """
    # Create a unique cache name based on file path
    file_hash = hashlib.md5(str(text_file).encode()).hexdigest()[:8]
    cache_name = f"{text_file.stem}_tokens_{file_hash}.pt"
    return text_file.parent / cache_name


def tokenize_parallel(text: str, tokenizer, chunk_size: int = 100_000) -> list[int]:
    """
    Tokenize text in parallel using batched encoding.

    The tokenizers library has built-in parallelism via encode_batch.
    We split text into chunks and process them in parallel.

    Args:
        text: Full text to tokenize
        tokenizer: The tokenizer instance
        chunk_size: Characters per chunk (not tokens)

    Returns:
        List of token IDs
    """
    # Split text into chunks at line boundaries
    lines = text.split('\n')

    # Group lines into batches
    batches = []
    current_batch = []
    current_size = 0

    for line in lines:
        current_batch.append(line)
        current_size += len(line)

        if current_size >= chunk_size:
            batches.append('\n'.join(current_batch))
            current_batch = []
            current_size = 0

    # Don't forget the last batch
    if current_batch:
        batches.append('\n'.join(current_batch))

    print(f"  Tokenizing {len(batches)} chunks in parallel...")

    # encode_batch uses multiple threads internally
    encoded_batches = tokenizer.encode_batch(batches)

    # Flatten all token IDs
    all_tokens = []
    for encoded in encoded_batches:
        all_tokens.extend(encoded.ids)

    return all_tokens


class TextDataset(Dataset):
    """
    Dataset that loads tokenized text and creates fixed-length chunks.

    Features:
    - Caches tokenized data to disk (fast loading after first run)
    - Parallel tokenization for large files
    - Memory-efficient: stores tokens as memory-mapped file option
    """

    def __init__(
        self,
        text_file: Path,
        context_length: int = 512,
        tokenizer_path: Path | None = None,
        use_cache: bool = True,
    ):
        """
        Args:
            text_file: Path to text file (train.txt or validation.txt)
            context_length: Length of each training sequence
            tokenizer_path: Path to tokenizer.json (uses default if None)
            use_cache: Whether to cache tokenized data to disk
        """
        self.context_length = context_length
        self.text_file = Path(text_file)

        # Load tokenizer
        self.tokenizer = load_tokenizer(tokenizer_path)

        # Check for cached tokens
        cache_path = get_cache_path(self.text_file)

        if use_cache and cache_path.exists():
            # Load from cache (fast!)
            print(f"Loading cached tokens from {cache_path.name}...")
            self.tokens = torch.load(cache_path)
            print(f"  Loaded {len(self.tokens):,} tokens from cache")
        else:
            # Tokenize from scratch
            print(f"Tokenizing {text_file}...")
            self.tokens = self._tokenize_file(text_file)

            # Save to cache for next time
            if use_cache:
                print(f"  Caching to {cache_path.name}...")
                torch.save(self.tokens, cache_path)

        print(f"  Total tokens: {len(self.tokens):,}")
        print(f"  Number of chunks: {len(self):,}")

    def _tokenize_file(self, text_file: Path) -> torch.Tensor:
        """Tokenize a text file with parallel processing."""
        # Read file
        print(f"  Reading {text_file.name}...")
        with open(text_file, "r", encoding="utf-8") as f:
            text = f.read()

        file_size_mb = len(text) / (1024 * 1024)
        print(f"  File size: {file_size_mb:.1f} MB")

        # Use parallel tokenization for large files
        if file_size_mb > 10:
            print(f"  Using parallel tokenization...")
            token_ids = tokenize_parallel(text, self.tokenizer)
        else:
            # Small file - just tokenize directly
            encoded = self.tokenizer.encode(text)
            token_ids = encoded.ids

        return torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        # How many chunks of context_length can we make?
        # We need context_length + 1 tokens for input + target
        return (len(self.tokens) - 1) // self.context_length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training example.

        Returns:
            input_ids: [context_length] tokens
            target_ids: [context_length] tokens (shifted by 1)
        """
        # Get chunk starting position
        start = idx * self.context_length
        end = start + self.context_length + 1  # +1 for target

        chunk = self.tokens[start:end]

        # Input is all but last token
        # Target is all but first token (shifted by 1)
        input_ids = chunk[:-1]    # [0, 1, 2, ..., n-1]
        target_ids = chunk[1:]    # [1, 2, 3, ..., n]

        return input_ids, target_ids

    def clear_cache(self):
        """Delete the cache file for this dataset."""
        cache_path = get_cache_path(self.text_file)
        if cache_path.exists():
            os.remove(cache_path)
            print(f"Deleted cache: {cache_path}")


def create_dataloaders(
    train_file: Path,
    val_file: Path,
    context_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
    use_cache: bool = True,
    tokenizer_path: Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_file: Path to training text file
        val_file: Path to validation text file
        context_length: Sequence length for training
        batch_size: Number of sequences per batch
        num_workers: Parallel data loading workers
        use_cache: Whether to cache tokenized data
        tokenizer_path: Path to tokenizer.json (uses default if None)

    Returns:
        train_loader, val_loader
    """
    train_dataset = TextDataset(train_file, context_length, tokenizer_path=tokenizer_path, use_cache=use_cache)
    val_dataset = TextDataset(val_file, context_length, tokenizer_path=tokenizer_path, use_cache=use_cache)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,      # Randomize order each epoch
        num_workers=num_workers,
        pin_memory=False,  # Disabled for MPS compatibility
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,     # No need to shuffle validation
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    from src.fetch_data.download import get_data_dir
    import time

    print("Testing TextDataset with caching...")
    print("=" * 50)

    # Try to find data
    data_dir = get_data_dir() / "tinystories_subset"
    if not data_dir.exists():
        data_dir = get_data_dir() / "tinystories"

    train_file = data_dir / "train.txt"

    if not train_file.exists():
        print(f"No data found at {train_file}")
        print("Run: uv run python -m src.fetch_data.download")
        exit(1)

    # First run - will tokenize and cache
    print("\nFirst run (tokenize + cache):")
    start = time.time()
    dataset = TextDataset(train_file, context_length=64, use_cache=True)
    first_time = time.time() - start
    print(f"Time: {first_time:.2f}s")

    # Second run - should load from cache
    print("\nSecond run (from cache):")
    start = time.time()
    dataset = TextDataset(train_file, context_length=64, use_cache=True)
    second_time = time.time() - start
    print(f"Time: {second_time:.2f}s")

    print(f"\nSpeedup: {first_time/second_time:.1f}x faster with cache!")

    print(f"\nDataset length: {len(dataset)}")

    # Get a sample
    input_ids, target_ids = dataset[0]
    print(f"\nSample input shape: {input_ids.shape}")
    print(f"Sample target shape: {target_ids.shape}")

    # Decode to show text
    input_text = dataset.tokenizer.decode(input_ids[:20].tolist())
    print(f"\nDecoded input: {input_text[:100]}...")

    print("\nDataset working!")
