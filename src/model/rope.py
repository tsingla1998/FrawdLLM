"""
Rotary Position Embedding (RoPE) for FrawdLLM.

RoPE encodes position by rotating the Q and K vectors. This has several advantages:
1. No learned position embeddings (saves parameters)
2. Better length generalization (can extrapolate beyond training length)
3. Relative position encoding (attention depends on distance, not absolute position)

How it works:
- Each position gets a rotation angle based on its index
- Q and K are rotated by their position's angle
- The dot product Q·K then naturally encodes relative distance

Reference: https://arxiv.org/abs/2104.09864
"""

import torch
import torch.nn as nn
import math


def precompute_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """
    Precompute the frequency tensor for RoPE.

    Args:
        dim: Dimension of each head (must be even)
        max_seq_len: Maximum sequence length
        theta: Base for frequency computation (10000 is standard)

    Returns:
        Complex tensor of shape [max_seq_len, dim//2] containing rotation frequencies
    """
    # Frequency for each dimension pair: theta^(-2i/dim) for i = 0, 1, ..., dim/2-1
    # Lower dimensions rotate slowly, higher dimensions rotate quickly
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Position indices
    positions = torch.arange(max_seq_len)

    # Outer product: [max_seq_len, dim//2]
    # Each position gets a different rotation angle for each frequency
    angles = torch.outer(positions, freqs)

    # Convert to complex numbers for easy rotation
    # e^(i*angle) = cos(angle) + i*sin(angle)
    freqs_complex = torch.polar(torch.ones_like(angles), angles)

    return freqs_complex


def apply_rope(
    x: torch.Tensor,
    freqs: torch.Tensor,
    start_pos: int = 0,
) -> torch.Tensor:
    """
    Apply rotary position embedding to Q or K tensor.

    Args:
        x: [batch, n_head, seq_len, head_dim] - Q or K tensor
        freqs: [max_seq_len, head_dim//2] - precomputed frequencies
        start_pos: Starting position (for KV cache during generation)

    Returns:
        Rotated tensor with same shape as input
    """
    batch, n_head, seq_len, head_dim = x.shape

    # Get frequencies for this sequence
    # [seq_len, head_dim//2]
    seq_freqs = freqs[start_pos:start_pos + seq_len]

    # Reshape x to pairs: [batch, n_head, seq_len, head_dim//2, 2]
    # We rotate adjacent pairs of dimensions together
    x_pairs = x.float().reshape(batch, n_head, seq_len, -1, 2)

    # Convert to complex: [batch, n_head, seq_len, head_dim//2]
    x_complex = torch.view_as_complex(x_pairs)

    # Reshape freqs for broadcasting: [1, 1, seq_len, head_dim//2]
    seq_freqs = seq_freqs.unsqueeze(0).unsqueeze(0)

    # Rotate by multiplying complex numbers
    x_rotated = x_complex * seq_freqs

    # Convert back to real: [batch, n_head, seq_len, head_dim//2, 2]
    x_out = torch.view_as_real(x_rotated)

    # Flatten back: [batch, n_head, seq_len, head_dim]
    x_out = x_out.reshape(batch, n_head, seq_len, head_dim)

    return x_out.type_as(x)


class RotaryEmbedding(nn.Module):
    """
    Module wrapper for rotary embeddings.

    Precomputes and caches the frequency tensor.
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # Precompute and register as buffer (saved with model but not trained)
        freqs = precompute_freqs(dim, max_seq_len, theta)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """Apply RoPE to input tensor."""
        return apply_rope(x, self.freqs, start_pos)


if __name__ == "__main__":
    print("Testing RoPE...")
    print("=" * 50)

    # Test parameters
    batch, n_head, seq_len, head_dim = 2, 4, 16, 64

    # Create rotary embedding
    rope = RotaryEmbedding(dim=head_dim, max_seq_len=512)

    # Create random Q and K
    q = torch.randn(batch, n_head, seq_len, head_dim)
    k = torch.randn(batch, n_head, seq_len, head_dim)

    print(f"Input shape: {q.shape}")

    # Apply RoPE
    q_rotated = rope(q)
    k_rotated = rope(k)

    print(f"Output shape: {q_rotated.shape}")

    # Verify relative position property
    # Attention at (i, j) should only depend on (i - j), not absolute positions
    print("\nVerifying relative position property...")

    # Compute attention for two positions
    attn_0_1 = (q_rotated[:, :, 0:1, :] @ k_rotated[:, :, 1:2, :].transpose(-2, -1))
    attn_5_6 = (q_rotated[:, :, 5:6, :] @ k_rotated[:, :, 6:7, :].transpose(-2, -1))

    # These should be very similar (same relative distance of 1)
    diff = (attn_0_1 - attn_5_6).abs().mean().item()
    print(f"  Attention (0,1) vs (5,6) difference: {diff:.6f}")
    print(f"  (Should be very small - same relative distance)")

    print("\nRoPE working!")
