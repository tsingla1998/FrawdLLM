# FrawdLLM

A learning-focused project to build a small language model from scratch.

## Goals

- Understand all phases of LLM training: Pre-training, SFT, RLHF
- Implement inference optimizations: KV-cache, paged attention, quantization
- Build something small but functional (~50M-125M parameters)

## Setup

```bash
uv sync
```

## Project Structure

```
FrawdLLM/
├── src/
│   ├── fetch_data/ # Data loading & tokenization
│   ├── model/      # GPT-2 and Llama architectures
│   ├── training/   # Training loops
│   └── inference/  # Inference optimizations
├── scripts/        # Training scripts
├── configs/        # Experiment configs
└── notebooks/      # Exploration notebooks
```

## Quick Start

```bash
# Download training data (10K subset for quick testing)
python -m src.fetch_data.download

# Train tokenizer
python -m src.fetch_data.tokenizer

# Train model (coming soon)
python scripts/train.py
```
