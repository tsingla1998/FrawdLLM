"""
Dataset for pre-tokenized OpenWebText data.

Loads tokenized data from binary files (memory-mapped for efficiency).
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class OpenWebTextDataset(Dataset):
    """
    Dataset for pre-tokenized text stored as binary numpy arrays.

    Memory-maps the file for efficient random access without loading all into RAM.
    """

    def __init__(
        self,
        data_path: Path,
        context_length: int = 1024,
        stride: int | None = None,
    ):
        """
        Args:
            data_path: Path to .bin file with tokenized data
            context_length: Sequence length for training
            stride: Step between sequences (default: context_length for no overlap)
        """
        self.context_length = context_length
        self.stride = stride or context_length

        # Memory-map the file (doesn't load into RAM)
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)

        # Calculate number of sequences
        # We need context_length + 1 tokens for each sequence (input + target)
        self.n_sequences = max(0, (self.n_tokens - context_length - 1) // self.stride + 1)

        print(f"Loaded {self.n_tokens:,} tokens from {data_path}")
        print(f"  Context length: {context_length}")
        print(f"  Sequences: {self.n_sequences:,}")

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a training sequence."""
        start = idx * self.stride
        end = start + self.context_length + 1

        # Get tokens
        tokens = torch.from_numpy(self.data[start:end].astype(np.int64))

        # Split into input and target
        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        return input_ids, target_ids


def create_openwebtext_dataloaders(
    data_dir: Path,
    context_length: int = 1024,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Create train and validation dataloaders for OpenWebText.

    Args:
        data_dir: Directory containing train.bin, val.bin, and meta.json
        context_length: Sequence length
        batch_size: Batch size
        num_workers: DataLoader workers

    Returns:
        train_loader, val_loader, meta
    """
    data_dir = Path(data_dir)

    # Load metadata
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)

    print(f"OpenWebText dataset:")
    print(f"  Train tokens: {meta['train_tokens']:,}")
    print(f"  Val tokens: {meta['val_tokens']:,}")
    print(f"  Vocab size: {meta['vocab_size']:,}")

    # Create datasets
    train_dataset = OpenWebTextDataset(
        data_dir / "train.bin",
        context_length=context_length,
    )

    val_dataset = OpenWebTextDataset(
        data_dir / "val.bin",
        context_length=context_length,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,  # Drop incomplete batches
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, meta


if __name__ == "__main__":
    # Test
    data_dir = Path(__file__).parent.parent.parent / "data" / "openwebtext"

    if not (data_dir / "train.bin").exists():
        print(f"Data not found at {data_dir}")
        print("Run: uv run python -m src.fetch_data.prepare_openwebtext")
        exit(1)

    train_loader, val_loader, meta = create_openwebtext_dataloaders(
        data_dir,
        context_length=1024,
        batch_size=4,
    )

    print(f"\nTrain batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")

    # Get a batch
    batch = next(iter(train_loader))
    input_ids, target_ids = batch

    print(f"\nBatch shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  target_ids: {target_ids.shape}")
