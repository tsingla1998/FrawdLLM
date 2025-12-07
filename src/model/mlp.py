"""
MLP (Multi-Layer Perceptron) for FrawdLLM.

This is the "feed-forward" part of the transformer block.
After attention lets tokens gather information from each other,
MLP lets each token process that information independently.

Structure:
    Input (768) → Expand (3072) → GELU → Shrink (768) → Output

The 4x expansion gives the model more "thinking room" before
compressing back to the original size.
"""

import torch
import torch.nn as nn

from .config import ModelConfig


class MLP(nn.Module):
    """
    Simple feed-forward network with GELU activation.

    Input:  [batch_size, seq_len, n_embd]
    Output: [batch_size, seq_len, n_embd]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Hidden dimension is 4x the embedding dimension
        # This is a common ratio used in most transformers
        hidden_dim = 4 * config.n_embd

        # Expand: 768 → 3072
        self.fc1 = nn.Linear(config.n_embd, hidden_dim)

        # Activation function
        self.act = nn.GELU()

        # Shrink: 3072 → 768
        self.fc2 = nn.Linear(hidden_dim, config.n_embd)

        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply MLP to each token independently.

        Args:
            x: [batch_size, seq_len, n_embd]

        Returns:
            [batch_size, seq_len, n_embd]
        """
        # Step 1: Expand
        # [batch, seq, 768] → [batch, seq, 3072]
        x = self.fc1(x)

        # Step 2: Non-linearity
        # [batch, seq, 3072] → [batch, seq, 3072] (same shape, different values)
        x = self.act(x)

        # Step 3: Shrink back
        # [batch, seq, 3072] → [batch, seq, 768]
        x = self.fc2(x)

        # Step 4: Dropout
        x = self.dropout(x)

        return x


if __name__ == "__main__":
    # Test the MLP module
    from .config import get_config

    print("Testing MLP...")
    print("=" * 50)

    config = get_config("tiny")
    hidden_dim = 4 * config.n_embd
    print(f"Config: n_embd={config.n_embd}, hidden_dim={hidden_dim}")

    mlp = MLP(config)

    # Count parameters
    num_params = sum(p.numel() for p in mlp.parameters())
    print(f"MLP parameters: {num_params:,}")

    # Test input: [batch=2, seq=8, n_embd=256]
    x = torch.randn(2, 8, config.n_embd)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    out = mlp(x)
    print(f"Output shape: {out.shape}")

    # Verify shapes match
    assert x.shape == out.shape, "Input and output shapes should match!"
    print("\nMLP working!")
