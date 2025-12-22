"""
Convert FrawdLLM checkpoint to HuggingFace format.

Usage:
    uv run python scripts/convert_to_hf.py --checkpoint checkpoints_100m_dpo/best.pt --output hf_model
"""

import argparse
import shutil
from pathlib import Path


def convert(checkpoint_path: str, output_dir: str, tokenizer_path: str | None = None):
    """Convert a FrawdLLM checkpoint to HuggingFace format."""
    import torch
    from src.model.hf_wrapper import FrawdLLMForCausalLM

    print(f"Loading checkpoint from {checkpoint_path}...")
    model = FrawdLLMForCausalLM.from_frawdllm_checkpoint(checkpoint_path)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Save in HuggingFace format
    print(f"Saving to {output_dir}...")
    model.save_pretrained_simple(output_dir)

    # Add auto_map to config.json for trust_remote_code
    import json
    config_path = Path(output_dir) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    config["auto_map"] = {
        "AutoConfig": "hf_wrapper.FrawdLLMConfig",
        "AutoModelForCausalLM": "hf_wrapper.FrawdLLMForCausalLM"
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print("Added auto_map to config.json")

    # Copy tokenizer if provided
    if tokenizer_path:
        tokenizer_src = Path(tokenizer_path)
        if tokenizer_src.exists():
            tokenizer_dst = Path(output_dir) / "tokenizer.json"
            shutil.copy(tokenizer_src, tokenizer_dst)
            print(f"Copied tokenizer to {tokenizer_dst}")

            # Create tokenizer_config.json for HuggingFace
            import json
            tokenizer_config = {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "bos_token": "<|bos|>",
                "eos_token": "<|eos|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "model_max_length": model.config.context_length,
            }
            with open(Path(output_dir) / "tokenizer_config.json", "w") as f:
                json.dump(tokenizer_config, f, indent=2)
            print("Created tokenizer_config.json")

    # Copy model files for trust_remote_code
    model_files = [
        "src/model/config.py",
        "src/model/gpt.py",
        "src/model/embeddings.py",
        "src/model/block.py",
        "src/model/attention.py",
        "src/model/mlp.py",
        "src/model/rope.py",
        "src/model/hf_wrapper.py",
    ]

    output_path = Path(output_dir)
    for f in model_files:
        src = Path(f)
        if src.exists():
            # Flatten to single directory for HF
            dst = output_path / src.name
            shutil.copy(src, dst)
            print(f"Copied {src.name}")

    # Create __init__.py that imports the wrapper
    init_content = '''"""FrawdLLM model for HuggingFace."""
from .hf_wrapper import FrawdLLMConfig, FrawdLLMForCausalLM

__all__ = ["FrawdLLMConfig", "FrawdLLMForCausalLM"]
'''
    with open(output_path / "__init__.py", "w") as f:
        f.write(init_content)

    # Fix imports in copied files (they use relative imports)
    fix_imports(output_path)

    print(f"\nConversion complete!")
    print(f"\nTo upload to HuggingFace:")
    print(f"  huggingface-cli login")
    print(f"  huggingface-cli upload tsingla1998/frawdllm-100m {output_dir}")
    print(f"\nTo load the model:")
    print(f'  from transformers import AutoModelForCausalLM')
    print(f'  model = AutoModelForCausalLM.from_pretrained("{output_dir}", trust_remote_code=True)')


def fix_imports(output_path: Path):
    """Ensure relative imports use . prefix for HuggingFace dynamic loading."""
    import re

    files_to_fix = ["gpt.py", "hf_wrapper.py", "block.py", "embeddings.py", "attention.py", "mlp.py"]

    for filename in files_to_fix:
        filepath = output_path / filename
        if not filepath.exists():
            continue

        with open(filepath) as f:
            content = f.read()

        # Ensure imports use . prefix (HuggingFace treats model dir as package)
        # Match "from config import" and change to "from .config import"
        content = re.sub(r"from (config|gpt|embeddings|block|attention|mlp|rope) import", r"from .\1 import", content)

        with open(filepath, "w") as f:
            f.write(content)

    print("Fixed imports for HuggingFace dynamic module loading")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FrawdLLM to HuggingFace format")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="hf_model", help="Output directory")
    parser.add_argument("--tokenizer", default="tokenizer_100m/tokenizer.json", help="Path to tokenizer.json")
    args = parser.parse_args()

    convert(args.checkpoint, args.output, args.tokenizer)
