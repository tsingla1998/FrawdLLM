"""
Test FrawdLLM model from HuggingFace.

Usage:
    # Test from HuggingFace
    uv run python scripts/test_hf_model.py

    # Test local hf_model directory
    uv run python scripts/test_hf_model.py --local
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast


def main(local: bool = False, repo_id: str = "tsingla98/frawdllm-100m"):
    # Load model
    if local:
        print("Loading from local hf_model/...")
        model_path = "hf_model"
    else:
        print(f"Loading from HuggingFace: {repo_id}...")
        model_path = repo_id

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # Load tokenizer
    if local:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="hf_model/tokenizer.json")
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(repo_id)

    # Set pad token if not set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 0

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Device: {device}")
    model = model.to(device)
    model.eval()

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print("\n" + "=" * 60)
    print("Testing FrawdLLM from HuggingFace")
    print("=" * 60)

    # Test prompts
    prompts = [
        "What is photosynthesis?",
        "Explain how computers work.",
        "What are the benefits of exercise?",
    ]

    for prompt in prompts:
        formatted = f"<|bos|><|user|>{prompt}<|assistant|>"
        # Don't add special tokens - they're already in the prompt
        inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)
        input_len = inputs.input_ids.shape[1]

        # Use our model's generate method (works better than HF's)
        with torch.no_grad():
            outputs = model.model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                temperature=0.8,
                top_k=50,
            )

        response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
        response = response.replace("<|eos|>", "").replace("<|pad|>", "").strip()

        print(f"\nQ: {prompt}")
        print(f"A: {response[:500]}")

    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Test local hf_model/ instead of HuggingFace")
    parser.add_argument("--repo", default="tsingla98/frawdllm-100m", help="HuggingFace repo ID")
    args = parser.parse_args()

    main(local=args.local, repo_id=args.repo)
