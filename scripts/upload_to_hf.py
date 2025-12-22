"""
Upload FrawdLLM to HuggingFace Hub.

Usage:
    uv run python scripts/upload_to_hf.py --checkpoint checkpoints_100m_dpo/best.pt --repo tsingla98/frawdllm-100m
"""

import argparse
import shutil
from pathlib import Path

import torch


def upload(checkpoint_path: str, repo_id: str, tokenizer_path: str = "tokenizer_100m/tokenizer.json"):
    """Upload model to HuggingFace Hub using push_to_hub()."""

    # Import our wrapper (this registers for AutoClass)
    from src.model.hf_wrapper import FrawdLLMForCausalLM, FrawdLLMConfig
    from src.model.config import ModelConfig

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config: ModelConfig = checkpoint["config"]

    # Create HF config
    hf_config = FrawdLLMConfig.from_model_config(model_config)

    # Create model and load weights
    model = FrawdLLMForCausalLM(hf_config)
    model.model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Push to Hub - this copies model code automatically
    print(f"Pushing to {repo_id}...")
    model.push_to_hub(repo_id)
    hf_config.push_to_hub(repo_id)

    # Upload tokenizer separately
    if tokenizer_path and Path(tokenizer_path).exists():
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=tokenizer_path,
            path_in_repo="tokenizer.json",
            repo_id=repo_id,
        )
        print("Uploaded tokenizer.json")

        # Create and upload tokenizer_config.json
        import json
        tokenizer_config = {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "bos_token": "<|bos|>",
            "eos_token": "<|eos|>",
            "pad_token": "<|pad|>",
            "unk_token": "<|unk|>",
            "model_max_length": model_config.context_length,
        }
        config_path = Path("tokenizer_config.json")
        with open(config_path, "w") as f:
            json.dump(tokenizer_config, f, indent=2)
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="tokenizer_config.json",
            repo_id=repo_id,
        )
        config_path.unlink()
        print("Uploaded tokenizer_config.json")

    print(f"\nDone! Model available at: https://huggingface.co/{repo_id}")
    print(f"\nTo load:")
    print(f'  from transformers import AutoModelForCausalLM')
    print(f'  model = AutoModelForCausalLM.from_pretrained("{repo_id}", trust_remote_code=True)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--repo", required=True, help="HuggingFace repo ID (e.g., username/model-name)")
    parser.add_argument("--tokenizer", default="tokenizer_100m/tokenizer.json", help="Path to tokenizer.json")
    args = parser.parse_args()

    upload(args.checkpoint, args.repo, args.tokenizer)
