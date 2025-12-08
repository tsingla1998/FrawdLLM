"""
Model configuration for FrawdLLM.

This module defines the hyperparameters that control model architecture.
We'll define multiple sizes to experiment with.

Learning Notes:
--------------
Key hyperparameters and their effects:

1. vocab_size: Size of tokenizer vocabulary
   - Must match your trained tokenizer
   - Larger = more memory for embedding table

2. n_embd (embedding dimension): Size of hidden representations
   - Larger = more expressive, but slower and more memory
   - GPT-2 small: 768, GPT-2 large: 1280, GPT-3: 12288

3. n_layer: Number of transformer blocks
   - More layers = deeper reasoning, but harder to train
   - GPT-2 small: 12, GPT-2 large: 36

4. n_head: Number of attention heads
   - Usually n_embd / n_head = 64 (head dimension)
   - More heads = more parallel attention patterns

5. context_length: Maximum sequence length
   - Longer = can process more text, but O(n²) memory for attention
   - GPT-2: 1024, GPT-3: 2048, modern models: 4096-128K

6. dropout: Regularization to prevent overfitting
   - 0.0 for small datasets (we need all the learning we can get)
   - 0.1-0.2 for larger datasets
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for FrawdLLM model."""

    # Vocabulary (must match tokenizer)
    vocab_size: int = 8192

    # Model dimensions
    n_embd: int = 768        # Embedding dimension
    n_layer: int = 12        # Number of transformer blocks
    n_head: int = 12         # Number of attention heads

    # Sequence length
    context_length: int = 512  # Maximum sequence length

    # Regularization
    dropout: float = 0.0     # Dropout probability (0 for small data)

    # Architecture choices (we'll implement both!)
    use_rope: bool = False   # Use Rotary Position Embeddings (Llama-style)
    use_rmsnorm: bool = False  # Use RMSNorm instead of LayerNorm (Llama-style)
    use_swiglu: bool = False   # Use SwiGLU activation (Llama-style)

    # Special token IDs (must match tokenizer)
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 3

    def __post_init__(self):
        """Validate configuration."""
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"

        self.head_dim = self.n_embd // self.n_head

    @property
    def num_parameters(self) -> int:
        """Estimate total number of parameters."""
        # Token embeddings: vocab_size * n_embd
        token_emb = self.vocab_size * self.n_embd

        # Position embeddings (if not using RoPE): context_length * n_embd
        pos_emb = 0 if self.use_rope else self.context_length * self.n_embd

        # Per transformer block:
        # - Attention: 4 * n_embd^2 (Q, K, V, O projections)
        # - MLP: 8 * n_embd^2 (up, down) or 12 * n_embd^2 (SwiGLU has gate)
        # - LayerNorms: 2 * n_embd (or 4 * n_embd with biases)
        mlp_factor = 12 if self.use_swiglu else 8
        per_block = 4 * self.n_embd**2 + mlp_factor * self.n_embd**2 + 4 * self.n_embd
        total_blocks = self.n_layer * per_block

        # Output projection (tied with token embeddings usually, so not counted)
        # Final layer norm: n_embd
        final_ln = self.n_embd

        return token_emb + pos_emb + total_blocks + final_ln


# Predefined configurations for different sizes
# These are designed to be trainable on different hardware

# ~10M parameters - For quick debugging on CPU/M3
# Can train in minutes on a laptop
FRAWDLLM_TINY = ModelConfig(
    vocab_size=8192,
    n_embd=256,
    n_layer=6,
    n_head=8,
    context_length=256,
    dropout=0.0,
)

# ~50M parameters - Good for learning on M3/single GPU
# Can train in hours on M3, generates reasonable text
FRAWDLLM_SMALL = ModelConfig(
    vocab_size=8192,
    n_embd=512,
    n_layer=8,
    n_head=8,
    context_length=512,
    dropout=0.0,
)

# ~125M parameters - Similar to GPT-2 small
# Needs GPU (AWS), generates good quality text
FRAWDLLM_BASE = ModelConfig(
    vocab_size=8192,
    n_embd=768,
    n_layer=12,
    n_head=12,
    context_length=1024,
    dropout=0.1,
)


# Llama-style variants (modern architecture)
FRAWDLLM_TINY_LLAMA = ModelConfig(
    vocab_size=8192,
    n_embd=256,
    n_layer=6,
    n_head=8,
    context_length=256,
    dropout=0.0,
    use_rope=True,
    use_rmsnorm=True,
    use_swiglu=True,
)

FRAWDLLM_SMALL_LLAMA = ModelConfig(
    vocab_size=8192,
    n_embd=512,
    n_layer=8,
    n_head=8,
    context_length=512,
    dropout=0.0,
    use_rope=True,
    use_rmsnorm=True,
    use_swiglu=True,
)

# ~100M parameters - Similar to GPT-2 Small but with modern architecture
# Uses RoPE for position encoding, allowing longer context at inference
FRAWDLLM_100M = ModelConfig(
    vocab_size=32000,       # Larger vocab for diverse data
    n_embd=768,
    n_layer=12,
    n_head=12,
    context_length=1024,    # Train on 1024, can extrapolate to 2048+
    dropout=0.1,
    use_rope=True,          # Rotary position embeddings
    use_rmsnorm=False,      # Keep LayerNorm for now
    use_swiglu=False,       # Keep GELU for now
)


def get_config(name: str) -> ModelConfig:
    """Get a predefined configuration by name."""
    configs = {
        "tiny": FRAWDLLM_TINY,
        "small": FRAWDLLM_SMALL,
        "base": FRAWDLLM_BASE,
        "tiny-llama": FRAWDLLM_TINY_LLAMA,
        "small-llama": FRAWDLLM_SMALL_LLAMA,
        "100m": FRAWDLLM_100M,
    }

    if name not in configs:
        raise ValueError(f"Unknown config: {name}. Available: {list(configs.keys())}")

    return configs[name]


if __name__ == "__main__":
    # Print parameter counts for each config
    print("FrawdLLM Model Configurations")
    print("=" * 50)

    for name in ["tiny", "small", "base", "tiny-llama", "small-llama"]:
        config = get_config(name)
        params = config.num_parameters
        print(f"\n{name}:")
        print(f"  Parameters: {params:,} ({params/1e6:.1f}M)")
        print(f"  Embedding dim: {config.n_embd}")
        print(f"  Layers: {config.n_layer}")
        print(f"  Heads: {config.n_head}")
        print(f"  Context: {config.context_length}")
        if config.use_rope:
            print(f"  Style: Llama (RoPE, RMSNorm, SwiGLU)")
        else:
            print(f"  Style: GPT-2 (learned pos, LayerNorm, GELU)")
