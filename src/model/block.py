"""
Transformer Block for FrawdLLM.

A transformer block combines:
1. Multi-head self-attention (tokens gather info from each other)
2. MLP (each token processes info independently)

With two important additions:
- LayerNorm: Keeps values stable during training
- Residual connections: Add input to output ("don't lose what you had")

Structure (Pre-LN, which is more stable):

    Input
      ↓
    ┌─────────────┐
    │  LayerNorm  │
    └─────────────┘
      ↓
    ┌─────────────┐
    │  Attention  │───────┐
    └─────────────┘       │ (residual)
      ↓                   │
      + ←─────────────────┘
      ↓
    ┌─────────────┐
    │  LayerNorm  │
    └─────────────┘
      ↓
    ┌─────────────┐
    │     MLP     │───────┐
    └─────────────┘       │ (residual)
      ↓                   │
      + ←─────────────────┘
      ↓
    Output
"""

import torch
import torch.nn as nn

from .config import ModelConfig
from .attention import CausalSelfAttention
from .mlp import MLP


class TransformerBlock(nn.Module):
    """
    One transformer block = Attention + MLP with norms and residuals.

    Input:  [batch_size, seq_len, n_embd]
    Output: [batch_size, seq_len, n_embd]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Layer norms (one before attention, one before MLP)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        # Attention and MLP
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply transformer block.

        Args:
            x: [batch_size, seq_len, n_embd]

        Returns:
            [batch_size, seq_len, n_embd]
        """
        # Attention with residual connection
        # x + attention(norm(x))
        # "Keep x, add attention's contribution"
        x = x + self.attn(self.ln1(x))

        # MLP with residual connection
        # x + mlp(norm(x))
        # "Keep x, add MLP's contribution"
        x = x + self.mlp(self.ln2(x))

        return x


if __name__ == "__main__":
    # Test the transformer block
    from .config import get_config

    print("Testing TransformerBlock...")
    print("=" * 50)

    config = get_config("tiny")
    print(f"Config: n_embd={config.n_embd}, n_head={config.n_head}, "
          f"n_layer={config.n_layer}")

    block = TransformerBlock(config)

    # Count parameters
    num_params = sum(p.numel() for p in block.parameters())
    print(f"Block parameters: {num_params:,}")

    # Test input: [batch=2, seq=8, n_embd=256]
    x = torch.randn(2, 8, config.n_embd)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    out = block(x)
    print(f"Output shape: {out.shape}")

    # Verify shapes match
    assert x.shape == out.shape, "Input and output shapes should match!"
    print("\nTransformerBlock working!")
