"""
DPO (Direct Preference Optimization) Dataset.

Loads preference data (prompt, chosen, rejected) and prepares for DPO training.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

from src.fetch_data.tokenizer import load_tokenizer

# Chat tokens (same as SFT)
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"


class DPODataset(Dataset):
    """
    Dataset for DPO training.

    Each example contains:
    - prompt_ids: tokenized prompt
    - chosen_ids: tokenized chosen response
    - rejected_ids: tokenized rejected response
    """

    def __init__(
        self,
        data_file: Path,
        tokenizer_path: Path | None = None,
        max_length: int = 512,
    ):
        self.max_length = max_length

        # Load tokenizer with chat tokens
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.tokenizer.add_special_tokens([USER_TOKEN, ASSISTANT_TOKEN])

        # Special token IDs
        self.pad_id = self.tokenizer.token_to_id("<|pad|>")
        self.bos_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")
        self.user_id = self.tokenizer.token_to_id(USER_TOKEN)
        self.assistant_id = self.tokenizer.token_to_id(ASSISTANT_TOKEN)

        # Load data
        print(f"Loading DPO data from {data_file}...")
        self.examples = []
        with open(data_file) as f:
            for line in f:
                if line.strip():
                    self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples):,} preference pairs")

        # Tokenize all examples
        print("Tokenizing examples...")
        self.tokenized = []

        for ex in tqdm(self.examples, desc="Tokenizing"):
            prompt = ex["prompt"]
            chosen = ex["chosen"]
            rejected = ex["rejected"]

            # Format: <|bos|><|user|>prompt<|assistant|>response<|eos|>
            prompt_tokens = (
                [self.bos_id, self.user_id] +
                self.tokenizer.encode(prompt, add_special_tokens=False).ids +
                [self.assistant_id]
            )

            chosen_tokens = (
                self.tokenizer.encode(chosen, add_special_tokens=False).ids +
                [self.eos_id]
            )

            rejected_tokens = (
                self.tokenizer.encode(rejected, add_special_tokens=False).ids +
                [self.eos_id]
            )

            # Truncate if needed (keep prompt, truncate responses)
            max_response_len = self.max_length - len(prompt_tokens)
            if len(chosen_tokens) > max_response_len:
                chosen_tokens = chosen_tokens[:max_response_len - 1] + [self.eos_id]
            if len(rejected_tokens) > max_response_len:
                rejected_tokens = rejected_tokens[:max_response_len - 1] + [self.eos_id]

            self.tokenized.append({
                "prompt": prompt_tokens,
                "chosen": chosen_tokens,
                "rejected": rejected_tokens,
            })

    def __len__(self) -> int:
        return len(self.tokenized)

    def __getitem__(self, idx: int) -> dict:
        """Get a preference pair."""
        item = self.tokenized[idx]

        # Full sequences
        chosen_full = item["prompt"] + item["chosen"]
        rejected_full = item["prompt"] + item["rejected"]

        # Pad to max_length
        chosen_padded = self._pad(chosen_full)
        rejected_padded = self._pad(rejected_full)

        # Create labels (for computing log probs on response only)
        # -100 = ignore in loss computation
        chosen_labels = self._create_labels(item["prompt"], item["chosen"])
        rejected_labels = self._create_labels(item["prompt"], item["rejected"])

        return {
            "chosen_input_ids": torch.tensor(chosen_padded, dtype=torch.long),
            "chosen_labels": torch.tensor(chosen_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rejected_padded, dtype=torch.long),
            "rejected_labels": torch.tensor(rejected_labels, dtype=torch.long),
        }

    def _pad(self, tokens: list[int]) -> list[int]:
        """Pad or truncate to max_length."""
        if len(tokens) >= self.max_length:
            return tokens[:self.max_length]
        return tokens + [self.pad_id] * (self.max_length - len(tokens))

    def _create_labels(self, prompt: list[int], response: list[int]) -> list[int]:
        """Create labels that mask the prompt (only compute loss on response)."""
        # Ignore prompt tokens, keep response tokens
        labels = [-100] * len(prompt) + response
        # Pad
        if len(labels) >= self.max_length:
            return labels[:self.max_length]
        return labels + [-100] * (self.max_length - len(labels))

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()


def create_dpo_dataloaders(
    data_file: Path,
    val_split: float = 0.1,
    tokenizer_path: Path | None = None,
    max_length: int = 512,
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader, int]:
    """Create train and validation dataloaders for DPO."""

    dataset = DPODataset(
        data_file,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
    )

    # Split
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Split: {train_size:,} train, {val_size:,} validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, dataset.get_vocab_size()


if __name__ == "__main__":
    # Test
    dpo_file = Path(__file__).parent.parent.parent / "data" / "dpo" / "preferences.jsonl"

    if not dpo_file.exists():
        print(f"DPO data not found at {dpo_file}")
        print("Run: uv run python -m src.fetch_data.generate_dpo")
        exit(1)

    dataset = DPODataset(dpo_file, max_length=256)
    print(f"\nDataset size: {len(dataset)}")

    item = dataset[0]
    print(f"\nSample shapes:")
    for k, v in item.items():
        print(f"  {k}: {v.shape}")
