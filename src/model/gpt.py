"""
Full GPT Model for FrawdLLM.

This is the complete model that:
1. Takes token IDs as input
2. Converts to embeddings (token + position)
3. Passes through N transformer blocks
4. Predicts the next token

Architecture:
    Token IDs [batch, seq]
        ↓
    Embeddings [batch, seq, n_embd]
        ↓
    Transformer Block × N
        ↓
    Final LayerNorm
        ↓
    Output Head → [batch, seq, vocab_size]
        ↓
    Logits (unnormalized probabilities for each vocab word)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .embeddings import Embeddings
from .block import TransformerBlock


class FrawdLLM(nn.Module):
    """
    The complete FrawdLLM model.

    Input:  token_ids [batch_size, seq_len]
    Output: logits [batch_size, seq_len, vocab_size]
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Token + position embeddings
        self.embeddings = Embeddings(config)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm (before output projection)
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output head: project from n_embd to vocab_size
        # This gives us a score for each possible next token
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embeddings and output head
        # This is a common trick that:
        # 1. Reduces parameters
        # 2. Makes sense: similar tokens should have similar embeddings AND predictions
        self.lm_head.weight = self.embeddings.token_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for better training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Forward pass through the model.

        Args:
            token_ids: [batch_size, seq_len] - input token IDs
            targets: [batch_size, seq_len] - target token IDs (for computing loss)

        Returns:
            logits: [batch_size, seq_len, vocab_size] - prediction scores
            loss: scalar tensor if targets provided, else None
        """
        # Step 1: Convert token IDs to embeddings
        # [batch, seq] → [batch, seq, n_embd]
        x = self.embeddings(token_ids)

        # Step 2: Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Step 3: Final layer norm
        x = self.ln_f(x)

        # Step 4: Project to vocabulary size
        # [batch, seq, n_embd] → [batch, seq, vocab_size]
        logits = self.lm_head(x)

        # Step 5: Compute loss if targets provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            # logits: [batch * seq, vocab_size]
            # targets: [batch * seq]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.config.pad_token_id,  # Don't compute loss on padding
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        token_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            token_ids: [batch_size, seq_len] - starting tokens (prompt)
            max_new_tokens: How many new tokens to generate
            temperature: Higher = more random, lower = more deterministic
            top_k: If set, only sample from top k most likely tokens

        Returns:
            [batch_size, seq_len + max_new_tokens] - original + generated tokens
        """
        for _ in range(max_new_tokens):
            # Crop to context length if needed
            context = token_ids[:, -self.config.context_length:]

            # Get predictions
            logits, _ = self.forward(context)

            # Take logits for the last position only
            # [batch, vocab_size]
            logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Optionally apply top-k filtering
            if top_k is not None:
                # Keep only top k values, set rest to -inf
                top_values, _ = torch.topk(logits, top_k, dim=-1)
                min_top_value = top_values[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_top_value,
                    torch.full_like(logits, float('-inf')),
                    logits,
                )

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            token_ids = torch.cat([token_ids, next_token], dim=1)

            # Stop if we generated EOS token
            if (next_token == self.config.eos_token_id).all():
                break

        return token_ids

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    from .config import get_config

    print("Testing FrawdLLM...")
    print("=" * 50)

    config = get_config("tiny")
    print(f"Config: vocab={config.vocab_size}, n_embd={config.n_embd}, "
          f"n_layer={config.n_layer}, n_head={config.n_head}")

    model = FrawdLLM(config)

    # Count parameters
    num_params = model.count_parameters()
    print(f"Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    # Test forward pass
    batch_size, seq_len = 2, 16
    token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    print(f"\nInput shape: {token_ids.shape}")

    logits, loss = model(token_ids, targets)

    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test generation
    prompt = torch.tensor([[config.bos_token_id]])  # Start with BOS
    generated = model.generate(prompt, max_new_tokens=10)
    print(f"\nGenerated shape: {generated.shape}")
    print(f"Generated tokens: {generated[0].tolist()}")

    print("\nFrawdLLM working!")
