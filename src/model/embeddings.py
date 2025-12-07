"""
Token and Position Embeddings for FrawdLLM.

This is the first layer of the model - converts token IDs into vectors
that the transformer can process.

Two lookup tables:
1. Token embeddings: WHAT the token is (vocab_size x n_embd)
2. Position embeddings: WHERE the token is (context_length x n_embd)

Final output = token_emb + pos_emb (just addition!)
"""

import torch
import torch.nn as nn

from .config import ModelConfig


class Embeddings(nn.Module):
    """
    Combined token + position embeddings.

    Input:  token_ids [batch_size, seq_len] - integers from tokenizer
    Output: vectors [batch_size, seq_len, n_embd] - dense representations
    """

    def __init__(self, config: ModelConfig):
        super().__init__()  # Initialize nn.Module tracking

        self.config = config

        # Token embedding table: one vector per vocabulary word
        # Shape: [vocab_size, n_embd] = [8192, 768]
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embedding table: one vector per position
        # Shape: [context_length, n_embd] = [512, 768]
        # This LIMITS our context window!
        self.pos_emb = nn.Embedding(config.context_length, config.n_embd)

        # Dropout for regularization (usually 0 for small datasets)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs to embeddings.

        Args:
            token_ids: [batch_size, seq_len] tensor of token IDs

        Returns:
            [batch_size, seq_len, n_embd] tensor of embeddings
        """
        batch_size, seq_len = token_ids.shape

        # Safety check: don't exceed context window
        if seq_len > self.config.context_length:
            raise ValueError(
                f"Sequence length {seq_len} exceeds context window "
                f"{self.config.context_length}"
            )

        # Step 1: Look up token embeddings
        # [batch_size, seq_len] -> [batch_size, seq_len, n_embd]
        tok_emb = self.token_emb(token_ids)

        # Step 2: Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=token_ids.device)

        # Step 3: Look up position embeddings
        # [seq_len] -> [seq_len, n_embd]
        pos_emb = self.pos_emb(positions)

        # Step 4: Add them together!
        # Broadcasting: [batch_size, seq_len, n_embd] + [seq_len, n_embd]
        # Result: [batch_size, seq_len, n_embd]
        embeddings = tok_emb + pos_emb

        # Step 5: Apply dropout (if any)
        embeddings = self.dropout(embeddings)

        return embeddings


if __name__ == "__main__":
    # Quick test to verify it works
    from .config import get_config

    print("Testing Embeddings...")
    print("=" * 50)

    # Use tiny config for testing
    config = get_config("tiny")
    print(f"Config: vocab={config.vocab_size}, n_embd={config.n_embd}, "
          f"context={config.context_length}")

    # Create embedding layer
    emb = Embeddings(config)

    # Count parameters
    num_params = sum(p.numel() for p in emb.parameters())
    print(f"Embedding parameters: {num_params:,}")

    # Test forward pass
    # Fake batch: 2 sequences of 4 tokens each
    token_ids = torch.tensor([
        [2, 531, 892, 12],   # Sequence 1
        [2, 100, 200, 3],    # Sequence 2
    ])

    print(f"\nInput shape: {token_ids.shape}")
    print(f"Input tokens: {token_ids.tolist()}")

    # Forward pass
    output = emb(token_ids)

    print(f"\nOutput shape: {output.shape}")
    print(f"Each token is now a {output.shape[-1]}-dimensional vector")

    # Show a snippet of the output
    print(f"\nFirst token's vector (first 10 dims):")
    print(f"  {output[0, 0, :10].tolist()}")

    print("\nEmbeddings working!")
