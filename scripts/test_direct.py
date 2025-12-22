"""Quick test of hf_model by direct import."""
import sys
import os

# Add parent of hf_model to path so we can import hf_model as a package
sys.path.insert(0, os.path.dirname(os.path.abspath('hf_model')))

from hf_model.hf_wrapper import FrawdLLMForCausalLM, FrawdLLMConfig
from safetensors.torch import load_file
import torch
import json

# Load config
with open('hf_model/config.json') as f:
    config_dict = json.load(f)

# Filter out non-model keys
model_keys = ['vocab_size', 'n_embd', 'n_layer', 'n_head', 'context_length',
              'dropout', 'use_rope', 'use_rmsnorm', 'use_swiglu',
              'pad_token_id', 'bos_token_id', 'eos_token_id']
config_dict = {k: v for k, v in config_dict.items() if k in model_keys}

config = FrawdLLMConfig(**config_dict)
model = FrawdLLMForCausalLM(config)

# Load weights
state_dict = load_file('hf_model/model.safetensors')
model.load_state_dict(state_dict, strict=False)
model.tie_weights()

print(f'Loaded: {sum(p.numel() for p in model.parameters()):,} parameters')

# Test
from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast(tokenizer_file='hf_model/tokenizer.json')

prompt = '<|bos|><|user|>What is the sun?<|assistant|>'
inputs = tokenizer(prompt, return_tensors='pt')
print(f'Input: {prompt}')

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        pad_token_id=0
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f'Output: {response}')
