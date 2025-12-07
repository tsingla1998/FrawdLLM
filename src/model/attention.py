"""
Multi-Head Self-Attention for FrawdLLM.

This is the core mechanism that lets tokens "look at" each other.
Each token creates:
  - Query (Q): "What am I looking for?"
  - Key (K):   "What do I contain?"
  - Value (V): "What information do I give?"

Attention score = how well Q matches K
Output = weighted sum of V based on attention scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .config import ModelConfig


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    "Causal" means tokens can only attend to past tokens, not future.
    This is required for language models (can't peek at what we're predicting!)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head  # e.g., 768/12 = 64

        # Linear projections to create Q, K, V
        # Each transforms [batch, seq, n_embd] -> [batch, seq, n_embd]
        # We do all three in one big matrix for efficiency, then split
        self.qkv_proj = nn.Linear(config.n_embd, 3 * config.n_embd)

        # Output projection: combines all heads back together
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        # Dropout for regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower triangular matrix
        # This prevents attending to future tokens
        # We register it as a buffer (saved with model, but not a parameter)
        mask = torch.tril(torch.ones(config.context_length, config.context_length))
        self.register_buffer("mask", mask.view(1, 1, config.context_length, config.context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head causal self-attention.

        Args:
            x: [batch_size, seq_len, n_embd] - input embeddings

        Returns:
            [batch_size, seq_len, n_embd] - attended embeddings
        """
        batch_size, seq_len, n_embd = x.shape

        # Step 1: Project to Q, K, V (all at once for efficiency)
        # [batch, seq, n_embd] -> [batch, seq, 3 * n_embd]
        qkv = self.qkv_proj(x)

        # Step 2: Split into Q, K, V
        # [batch, seq, 3 * n_embd] -> 3 x [batch, seq, n_embd]
        q, k, v = qkv.chunk(3, dim=-1)

        # Step 3: Reshape for multi-head attention
        # [batch, seq, n_embd] -> [batch, n_head, seq, head_dim]
        # Example: [32, 512, 768] -> [32, 12, 512, 64]
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        # Step 4: Compute attention scores
        # Q @ K^T: [batch, n_head, seq, head_dim] @ [batch, n_head, head_dim, seq]
        #        = [batch, n_head, seq, seq]
        # Each (i,j) entry = "how much should position i attend to position j?"
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 5: Apply causal mask (prevent attending to future)
        # Mask is 1 for allowed positions, 0 for disallowed
        # We set disallowed positions to -inf so softmax gives 0
        attn_scores = attn_scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )

        # Step 6: Softmax to get attention weights (probabilities)
        # [batch, n_head, seq, seq] - each row sums to 1
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Step 7: Apply attention to values
        # [batch, n_head, seq, seq] @ [batch, n_head, seq, head_dim]
        # = [batch, n_head, seq, head_dim]
        out = attn_weights @ v

        # Step 8: Reshape back: combine all heads
        # [batch, n_head, seq, head_dim] -> [batch, seq, n_head, head_dim] -> [batch, seq, n_embd]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, n_embd)

        # Step 9: Final output projection
        out = self.out_proj(out)
        out = self.resid_dropout(out)

        return out


if __name__ == "__main__":
    # Test the attention module
    from .config import get_config

    print("Testing CausalSelfAttention...")
    print("=" * 50)

    config = get_config("tiny")
    print(f"Config: n_embd={config.n_embd}, n_head={config.n_head}, "
          f"head_dim={config.head_dim}")

    attn = CausalSelfAttention(config)

    # Count parameters
    num_params = sum(p.numel() for p in attn.parameters())
    print(f"Attention parameters: {num_params:,}")

    # Test input: [batch=2, seq=8, n_embd=256]
    x = torch.randn(2, 8, config.n_embd)
    print(f"\nInput shape: {x.shape}")

    # Forward pass
    out = attn(x)
    print(f"Output shape: {out.shape}")

    # Verify shapes match
    assert x.shape == out.shape, "Input and output shapes should match!"
    print("\nAttention working!")
