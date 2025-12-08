"""
SFT (Supervised Fine-Tuning) Dataset for FrawdLLM.

Converts instruction-response pairs into training examples.

Format:
    Input:  <|bos|><|user|>Write a story...<|assistant|>Once upon a time...<|eos|>
    Target: (shifted by 1 for next-token prediction)

Key difference from PT:
- Data is structured as instruction-response pairs
- Optional: mask loss on instruction tokens (only learn from responses)
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizers import Tokenizer
from tqdm import tqdm

from src.fetch_data.tokenizer import load_tokenizer


# Chat tokens for SFT
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"


def add_chat_tokens(tokenizer: Tokenizer) -> tuple[Tokenizer, int, int]:
    """
    Add chat tokens to an existing tokenizer.

    Args:
        tokenizer: The base tokenizer

    Returns:
        tokenizer: Updated tokenizer
        user_token_id: ID of <|user|> token
        assistant_token_id: ID of <|assistant|> token
    """
    # Check if tokens already exist
    user_id = tokenizer.token_to_id(USER_TOKEN)
    assistant_id = tokenizer.token_to_id(ASSISTANT_TOKEN)

    if user_id is None or assistant_id is None:
        # Add new tokens
        num_added = tokenizer.add_special_tokens([USER_TOKEN, ASSISTANT_TOKEN])
        print(f"Added {num_added} chat tokens to tokenizer")
        user_id = tokenizer.token_to_id(USER_TOKEN)
        assistant_id = tokenizer.token_to_id(ASSISTANT_TOKEN)

    return tokenizer, user_id, assistant_id


def format_chat(instruction: str, response: str, tokenizer: Tokenizer) -> list[int]:
    """
    Format an instruction-response pair for training.

    Format: <|bos|><|user|>{instruction}<|assistant|>{response}<|eos|>

    Args:
        instruction: The user instruction
        response: The assistant response
        tokenizer: Tokenizer with chat tokens added

    Returns:
        List of token IDs
    """
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    user_id = tokenizer.token_to_id(USER_TOKEN)
    assistant_id = tokenizer.token_to_id(ASSISTANT_TOKEN)

    # Encode instruction and response separately (without special tokens)
    instruction_tokens = tokenizer.encode(instruction, add_special_tokens=False).ids
    response_tokens = tokenizer.encode(response, add_special_tokens=False).ids

    # Build the full sequence
    # <|bos|><|user|>{instruction}<|assistant|>{response}<|eos|>
    tokens = (
        [bos_id, user_id] +
        instruction_tokens +
        [assistant_id] +
        response_tokens +
        [eos_id]
    )

    return tokens


class SFTDataset(Dataset):
    """
    Dataset for SFT training.

    Loads instruction-response pairs from JSONL file and converts to training examples.
    """

    def __init__(
        self,
        data_file: Path,
        tokenizer_path: Path | None = None,
        max_length: int = 512,
        mask_instruction: bool = False,
    ):
        """
        Args:
            data_file: Path to JSONL file with instruction/response pairs
            tokenizer_path: Path to tokenizer.json
            max_length: Maximum sequence length (truncate longer sequences)
            mask_instruction: If True, don't compute loss on instruction tokens
        """
        self.max_length = max_length
        self.mask_instruction = mask_instruction

        # Load tokenizer and add chat tokens
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.tokenizer, self.user_id, self.assistant_id = add_chat_tokens(self.tokenizer)

        # Store special token IDs
        self.pad_id = self.tokenizer.token_to_id("<|pad|>")
        self.bos_id = self.tokenizer.token_to_id("<|bos|>")
        self.eos_id = self.tokenizer.token_to_id("<|eos|>")

        # Load data
        self.examples = []
        print(f"Loading SFT data from {data_file}...")

        with open(data_file, "r") as f:
            for line in f:
                if line.strip():
                    example = json.loads(line)
                    self.examples.append(example)

        print(f"Loaded {len(self.examples):,} examples")

        # Tokenize all examples
        print("Tokenizing examples...")
        self.tokenized = []
        self.instruction_lengths = []  # For loss masking

        for ex in tqdm(self.examples, desc="Tokenizing"):
            tokens = format_chat(ex["instruction"], ex["response"], self.tokenizer)

            # Track where instruction ends (for loss masking)
            instruction_tokens = self.tokenizer.encode(
                ex["instruction"], add_special_tokens=False
            ).ids
            # instruction_length = <|bos|> + <|user|> + instruction + <|assistant|>
            instruction_len = 2 + len(instruction_tokens) + 1

            # Truncate if needed
            if len(tokens) > max_length:
                tokens = tokens[:max_length - 1] + [self.eos_id]

            self.tokenized.append(tokens)
            self.instruction_lengths.append(min(instruction_len, len(tokens)))

        # Calculate stats
        lengths = [len(t) for t in self.tokenized]
        print(f"  Average length: {sum(lengths) / len(lengths):.1f} tokens")
        print(f"  Max length: {max(lengths)} tokens")
        truncated = sum(1 for l in lengths if l >= max_length)
        print(f"  Truncated: {truncated:,} examples ({100*truncated/len(lengths):.1f}%)")

    def __len__(self) -> int:
        return len(self.tokenized)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Get a training example.

        Returns:
            input_ids: [max_length-1] - input tokens
            target_ids: [max_length-1] - target tokens (shifted by 1)
            loss_mask: [max_length-1] - 1 for tokens to compute loss on, 0 otherwise
        """
        tokens = self.tokenized[idx]
        instruction_len = self.instruction_lengths[idx]

        # Pad to max_length
        padding_needed = self.max_length - len(tokens)
        if padding_needed > 0:
            tokens = tokens + [self.pad_id] * padding_needed

        tokens = torch.tensor(tokens, dtype=torch.long)

        # Input is all but last, target is all but first
        input_ids = tokens[:-1]
        target_ids = tokens[1:]

        # Create loss mask
        loss_mask = torch.ones_like(target_ids, dtype=torch.float)

        if self.mask_instruction:
            # Don't compute loss on instruction tokens
            # instruction_len - 1 because we shifted by 1
            loss_mask[:instruction_len - 1] = 0.0

        # Don't compute loss on padding
        loss_mask[target_ids == self.pad_id] = 0.0

        return input_ids, target_ids, loss_mask

    def get_vocab_size(self) -> int:
        """Get the vocabulary size (including added tokens)."""
        return self.tokenizer.get_vocab_size()


def create_sft_dataloaders(
    train_file: Path,
    val_file: Path | None = None,
    val_split: float = 0.1,
    tokenizer_path: Path | None = None,
    max_length: int = 512,
    batch_size: int = 32,
    mask_instruction: bool = False,
) -> tuple[DataLoader, DataLoader, int]:
    """
    Create train and validation dataloaders for SFT.

    Args:
        train_file: Path to training JSONL file
        val_file: Path to validation JSONL file (if None, split from train)
        val_split: Fraction to use for validation if val_file is None
        tokenizer_path: Path to tokenizer.json
        max_length: Maximum sequence length
        batch_size: Batch size
        mask_instruction: If True, don't compute loss on instruction tokens

    Returns:
        train_loader, val_loader, vocab_size
    """
    # Load full dataset
    full_dataset = SFTDataset(
        train_file,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        mask_instruction=mask_instruction,
    )

    vocab_size = full_dataset.get_vocab_size()

    if val_file is not None:
        # Separate validation file
        train_dataset = full_dataset
        val_dataset = SFTDataset(
            val_file,
            tokenizer_path=tokenizer_path,
            max_length=max_length,
            mask_instruction=mask_instruction,
        )
    else:
        # Split training data
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"Split: {train_size:,} train, {val_size:,} validation")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return train_loader, val_loader, vocab_size


if __name__ == "__main__":
    # Test the dataset
    sft_file = Path(__file__).parent.parent.parent / "data" / "sft" / "instructions.jsonl"

    if not sft_file.exists():
        print(f"SFT data not found at {sft_file}")
        print("Run: uv run python -m src.fetch_data.generate_sft")
        exit(1)

    print("Testing SFTDataset...")
    print("=" * 50)

    dataset = SFTDataset(sft_file, max_length=256, mask_instruction=True)

    print(f"\nDataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.get_vocab_size()}")

    # Get a sample
    input_ids, target_ids, loss_mask = dataset[0]
    print(f"\nSample shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  target_ids: {target_ids.shape}")
    print(f"  loss_mask: {loss_mask.shape}")
    print(f"  loss_mask sum: {loss_mask.sum().item():.0f} / {len(loss_mask)} tokens")

    # Decode and show
    print(f"\nDecoded sample:")
    decoded = dataset.tokenizer.decode(input_ids.tolist())
    print(decoded[:500])

    print("\n" + "=" * 50)
    print("SFTDataset working!")
