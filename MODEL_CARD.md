---
license: mit
language:
- en
tags:
- text-generation
- pytorch
- causal-lm
- from-scratch
datasets:
- openwebtext
library_name: transformers
pipeline_tag: text-generation
---

# FrawdLLM 100M

A 109M parameter language model trained from scratch for learning purposes.

## Model Description

FrawdLLM is a GPT-style decoder-only transformer trained through the full LLM pipeline:

1. **Pre-training (PT)**: Trained on OpenWebText for next-token prediction
2. **Supervised Fine-tuning (SFT)**: Fine-tuned on ~39K instruction-response pairs
3. **Direct Preference Optimization (DPO)**: Aligned using ~15K preference pairs ranked by Claude

### Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | 109M |
| Layers | 12 |
| Hidden Size | 768 |
| Attention Heads | 12 |
| Context Length | 1024 |
| Vocab Size | 32,000 |
| Position Encoding | RoPE |

## Usage

```python
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "tsingla98/frawdllm-100m",
    trust_remote_code=True
)
tokenizer = PreTrainedTokenizerFast.from_pretrained("tsingla98/frawdllm-100m")

# Format prompt with chat template
prompt = "<|bos|><|user|>What is photosynthesis?<|assistant|>"
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

# Generate
with torch.no_grad():
    output = model.model.generate(
        inputs.input_ids,
        max_new_tokens=150,
        temperature=0.8,
        top_k=50,
    )

response = tokenizer.decode(output[0], skip_special_tokens=False)
print(response)
```

## Training Details

### Pre-training
- **Data**: OpenWebText (8M documents)
- **Tokens**: ~2B tokens
- **Hardware**: Modal A100 GPU
- **Duration**: ~24 hours

### SFT
- **Data**: 39K instruction-response pairs generated with Claude
- **Categories**: Explanation, how-to, comparison, analysis, creative, factual, advice, summarization
- **Epochs**: 3
- **Final Loss**: 2.38

### DPO
- **Data**: 15K preference pairs
- **Ranking**: Claude Haiku
- **Beta**: 0.1
- **Epochs**: 3
- **Final Accuracy**: 63% (validation)

## Limitations

This is a small model (109M params) trained primarily for learning purposes. It:

- May generate incorrect or nonsensical information
- Has limited knowledge and reasoning capabilities
- Should not be used for production applications
- May occasionally produce repetitive text

## Intended Use

This model is intended for:
- Learning about LLM training pipelines
- Experimenting with inference optimizations
- Understanding transformer architectures

## Links

- **GitHub**: [github.com/tsingla1998/FrawdLLM](https://github.com/tsingla1998/FrawdLLM)
- **Training Code**: See the `scripts/` directory in the GitHub repo

## License

MIT
