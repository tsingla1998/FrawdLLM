from typing import Optional
from tokenizers import Tokenizer
from functools import cache
from dataclasses import dataclass

import torch
import math

EMBEDDINGS_WEIGHT_KEY = 'embeddings.token_emb.weight'

LN1_WEIGHT_FORMAT_PRE_ATTN_KEY = 'blocks.{}.ln1.weight'
LN1_BIAS_FORMAT_PRE_ATTN_KEY = 'blocks.{}.ln1.bias'

LN2_WEIGHT_FORMAT_PRE_ATTN_KEY = 'blocks.{}.ln2.weight'
LN2_BIAS_FORMAT_PRE_ATTN_KEY = 'blocks.{}.ln2.bias'

LNF_WEIGHT_FORMAT_KEY = 'ln_f.weight'
LNF_BIAS_FORMAT_KEY = 'ln_f.bias'

LM_HEAD_WEIGHT_KEY = 'lm_head.weight'

QKV_WEIGHTS_KEY = 'blocks.{}.attn.qkv_proj.weight'
QKV_BIAS_KEY = 'blocks.{}.attn.qkv_proj.bias'

OUTPUT_PROJ_BIAS_KEY = 'blocks.{}.attn.out_proj.bias'
OUTPUT_PROJ_WEIGHT_KEY = 'blocks.{}.attn.out_proj.weight'

FC1_WEIGHT_KEY = 'blocks.{}.mlp.fc1.weight'
FC1_BIAS_KEY = 'blocks.{}.mlp.fc1.bias'

FC2_WEIGHT_KEY = 'blocks.{}.mlp.fc2.weight'
FC2_BIAS_KEY = 'blocks.{}.mlp.fc2.bias'

STOP_TOKEN_ID = 3

EPSILON = 1e-5
N_HEADS = 12
N_LAYERS = 12
HEAD_DIM = 64
EMBEDDINGS_DIM = N_HEADS * HEAD_DIM
ROPE_THETA = 10000.0
TEMPERATURE = 0.5
TOP_P = 0.9

MAX_OUTPUT_TOKENS = 300

@dataclass
class KVCache:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    seq_len: int
    layer_num: int

@cache
def get_tokenizer() -> Tokenizer:
    return Tokenizer.from_file("tokenizer_100m/tokenizer.json")

def format_prompt(user_message: str) -> str:
    return f"<|bos|><|user|>{user_message}<|assistant|>"

@cache
def _get_weights_dict() -> dict[str, torch.Tensor]:
    return torch.load("checkpoints_100m_dpo/best.pt", map_location="cpu", weights_only=False)["model_state_dict"]

@cache
def get_weights_tensor(key: str) -> torch.Tensor:
    d = _get_weights_dict()
    if key not in d:
        raise ValueError(f'{key} is not in weights')
    return d[key]

def get_tokens_for_prompt(prompt: str) -> torch.Tensor:
    return torch.tensor(get_tokenizer().encode(prompt, add_special_tokens=False).ids)

def get_embeddings_for_selection(selection: torch.Tensor) -> torch.Tensor:
    return get_weights_tensor(EMBEDDINGS_WEIGHT_KEY)[selection]

@cache
def get_rope_freqs(seq_len: int) -> torch.Tensor:
    """
    Precompute rotation frequencies for RoPE.
    Returns: [seq_len, HEAD_DIM/2] tensor of angles
    """
    # Frequency for each dimension pair: 1 / (theta ^ (2i / dim))
    # i = 0, 1, 2, ..., HEAD_DIM/2 - 1
    dim_indices = torch.arange(0, HEAD_DIM, 2).float()  # [0, 2, 4, ..., 62]
    freqs = 1.0 / (ROPE_THETA ** (dim_indices / HEAD_DIM))  # [32] frequencies

    # Position indices
    positions = torch.arange(seq_len).float()  # [0, 1, 2, ..., seq_len-1]

    # Outer product: each position gets an angle for each frequency
    # angles[p, i] = position_p * freq_i
    angles = torch.outer(positions, freqs)  # [seq_len, 32]

    return angles

def apply_rope(x: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding to Q or K.
    x: [N_HEADS, seq_len, HEAD_DIM]
    angles: [seq_len, HEAD_DIM/2]
    Returns: rotated x with same shape
    """
    # Split into even and odd dimensions (pairs)
    x_even = x[..., 0::2]  # [N_HEADS, seq_len, 32]
    x_odd = x[..., 1::2]   # [N_HEADS, seq_len, 32]

    # Compute sin and cos (broadcast angles to match x shape)
    cos = torch.cos(angles)  # [seq_len, 32]
    sin = torch.sin(angles)  # [seq_len, 32]

    # Rotation formula (complex multiplication):
    # (x_even + i*x_odd) * (cos + i*sin) =
    #   (x_even*cos - x_odd*sin) + i*(x_even*sin + x_odd*cos)
    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot = x_even * sin + x_odd * cos

    # Interleave back: [even0, odd0, even1, odd1, ...]
    out = torch.stack([x_even_rot, x_odd_rot], dim=-1)  # [N_HEADS, seq_len, 32, 2]
    out = out.view(x.shape)  # [N_HEADS, seq_len, HEAD_DIM]

    return out

def process_attention(layer_num: int, x: torch.Tensor, kv_cache: Optional[KVCache]) -> tuple[torch.Tensor, KVCache]:
    # Attention
    if kv_cache:
        x = x[-1:]
        original_seq_len = kv_cache.seq_len + 1
    else:
        original_seq_len = x.shape[0]

    gamma = get_weights_tensor(LN1_WEIGHT_FORMAT_PRE_ATTN_KEY.format(layer_num))
    beta = get_weights_tensor(LN1_BIAS_FORMAT_PRE_ATTN_KEY.format(layer_num))
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    output = (x - mean) / (std + EPSILON)
    output = output * gamma + beta
    w_qkv = get_weights_tensor(QKV_WEIGHTS_KEY.format(layer_num))
    seq_len = x.shape[0]
    qkv = output @ w_qkv.T
    qkv_bias = get_weights_tensor(QKV_BIAS_KEY.format(layer_num))
    qkv += qkv_bias
    q, k, v = qkv.chunk(3, dim=-1)
    q_batched = q.view(seq_len, N_HEADS, HEAD_DIM).transpose(0, 1)  # [N_HEADS, seq_len, HEAD_DIM]
    k_batched = k.view(seq_len, N_HEADS, HEAD_DIM).transpose(0, 1)  # [N_HEADS, seq_len, HEAD_DIM]
    v_batched = v.view(seq_len, N_HEADS, HEAD_DIM).transpose(0, 1)  # [N_HEADS, seq_len, HEAD_DIM]
    
    # Apply RoPE to Q and K (not V)
    if kv_cache:
      angles = get_rope_freqs(original_seq_len)[-1:]
    else:
      angles = get_rope_freqs(original_seq_len)

    q_batched = apply_rope(q_batched, angles)
    k_batched = apply_rope(k_batched, angles)
    

    if kv_cache:
        # Use kv cache.        [seq_len, N_HEADS, HEAD_DIM] + [seq_len, N_HEADS, HEAD_DIM] 
        k_batched = torch.cat([kv_cache.k_cache, k_batched.transpose(0, 1)]).transpose(0, 1)  # [N_HEADS, seq_len, HEAD_DIM]
        v_batched = torch.cat([kv_cache.v_cache, v_batched.transpose(0, 1)]).transpose(0, 1)  # [N_HEADS, seq_len, HEAD_DIM]

                            #      [seq_len, N_HEADS, HEAD_DIM]
    new_kv_cache = KVCache(k_cache=k_batched.transpose(0, 1), v_cache=v_batched.transpose(0, 1), seq_len=original_seq_len, layer_num=layer_num)
    res_batched = q_batched @ k_batched.transpose(-2, -1)

    # We only need this for og prompt, last token can attend to all previous tokens
    if not kv_cache:
        mask = torch.triu(torch.ones([original_seq_len, original_seq_len]), diagonal=1) * -1e9
        res_batched += mask

    res_batched /= math.sqrt(HEAD_DIM)
    res_batched = res_batched.softmax(dim=-1)
    out = res_batched @ v_batched
    out = out.transpose(0, 1).contiguous().view(seq_len, EMBEDDINGS_DIM)
    output_proj_weights = get_weights_tensor(OUTPUT_PROJ_WEIGHT_KEY.format(layer_num))
    output_proj_bias = get_weights_tensor(OUTPUT_PROJ_BIAS_KEY.format(layer_num))
    out = out @ output_proj_weights.T + output_proj_bias + x
    return out, new_kv_cache


def process_mlp(layer_num: int, x: torch.Tensor) -> torch.Tensor:
    gamma = get_weights_tensor(LN2_WEIGHT_FORMAT_PRE_ATTN_KEY.format(layer_num))
    beta = get_weights_tensor(LN2_BIAS_FORMAT_PRE_ATTN_KEY.format(layer_num))
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    output = (x - mean) / (std + EPSILON)
    output = output * gamma + beta
    fc1_weights = get_weights_tensor(FC1_WEIGHT_KEY.format(layer_num))
    fc1_bias = get_weights_tensor(FC1_BIAS_KEY.format(layer_num))
    fc2_weights = get_weights_tensor(FC2_WEIGHT_KEY.format(layer_num))
    fc2_bias = get_weights_tensor(FC2_BIAS_KEY.format(layer_num))
    output = output @ fc1_weights.T + fc1_bias
    output = torch.nn.functional.gelu(output)
    output = output @ fc2_weights.T + fc2_bias
    return output + x

def process_layer(layer_num: int, x: torch.Tensor, kv_cache: KVCache) -> tuple[torch.Tensor, KVCache]:
    o, kv_cache = process_attention(layer_num, x, kv_cache)
    o = process_mlp(layer_num, o)
    return o, kv_cache

def process_final_layer(x: torch.Tensor) -> torch.Tensor:
    gamma = get_weights_tensor(LNF_WEIGHT_FORMAT_KEY)
    beta = get_weights_tensor(LNF_BIAS_FORMAT_KEY)
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    output = (x - mean) / (std + EPSILON)
    output = output * gamma + beta

    lm_head = get_weights_tensor(LM_HEAD_WEIGHT_KEY)
    return output @ lm_head.T

def main() -> None:
    # prompt = input("Prompt?")
    prompt = "Hello, World!"
    prompt = format_prompt(prompt)
    tokens = get_tokens_for_prompt(prompt)
    output_tokens = []
    kv_caches: list[Optional[KVCache]] = [None] * N_LAYERS
    for _ in range(MAX_OUTPUT_TOKENS):
        embeddings_for_prompt = get_embeddings_for_selection(tokens)
        x = embeddings_for_prompt
        for layer_num in range(N_LAYERS):
            x, kv_cache = process_layer(layer_num, x, kv_caches[layer_num])
            kv_caches[layer_num] = kv_cache
        x = process_final_layer(x)

        # TEMPERATURE
        logits = x[-1] / TEMPERATURE
        probs = logits.softmax(dim=-1)
        
        # TOP-P 
        sorted_probs, sorted_indicies = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=0)

        cum_probs_keep = cum_probs <= TOP_P
        cum_probs_keep[0] = True # Always keep the first token since it could be higher than TOP_P and would get excluded by the logic above
        
        sorted_probs[~cum_probs_keep] = 0

        next_token = sorted_indicies[torch.multinomial(sorted_probs, 1).item()]
        if next_token == STOP_TOKEN_ID:
            break
        tokens = torch.cat([tokens, torch.tensor([next_token])])
        output_tokens.append(next_token)
    
    output_words = get_tokenizer().decode(output_tokens)
    print(output_words)


if __name__ == "__main__":
    main()
    # print(_get_weights_dict().keys())