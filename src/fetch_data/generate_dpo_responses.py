"""
Phase 1: Generate response pairs from SFT model for DPO.

Generates 2 responses per prompt and saves them for later ranking.

Usage:
    uv run python -m src.fetch_data.generate_dpo_responses --num-examples 15000
"""

import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from src.model.gpt import FrawdLLM
from src.fetch_data.tokenizer import load_tokenizer
from src.training.sft_dataset import USER_TOKEN, ASSISTANT_TOKEN

# Paths
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "dpo_100m"
OUTPUT_FILE = OUTPUT_DIR / "response_pairs.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "gen_progress.json"

SFT_CHECKPOINT = Path(__file__).parent.parent.parent / "checkpoints_100m_sft" / "best.pt"
TOKENIZER_PATH = Path(__file__).parent.parent.parent / "tokenizer_100m" / "tokenizer.json"


def load_sft_model(device: str = "cpu"):
    """Load the SFT model."""
    print(f"Loading SFT model from {SFT_CHECKPOINT}...")

    checkpoint = torch.load(SFT_CHECKPOINT, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = FrawdLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer(TOKENIZER_PATH)
    tokenizer.add_special_tokens([USER_TOKEN, ASSISTANT_TOKEN])

    print(f"Loaded: {model.count_parameters():,} parameters")
    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 200) -> str:
    """Generate a single response."""
    formatted = f"<|bos|><|user|>{prompt}<|assistant|>"
    input_ids = tokenizer.encode(formatted, add_special_tokens=False).ids
    input_len = len(input_ids)
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=0.9,
            top_k=50,
        )

    response = tokenizer.decode(output[0][input_len:].tolist())
    response = response.replace("<|eos|>", "").replace("<|pad|>", "").strip()
    return response


def load_prompts(num_examples: int, seed: int = 42) -> list[str]:
    """Load prompts from SFT data."""
    sft_file = Path(__file__).parent.parent.parent / "data" / "sft_100m" / "instructions.jsonl"

    prompts = []
    with open(sft_file) as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                prompts.append(example["instruction"])

    random.seed(seed)
    random.shuffle(prompts)
    return prompts[:num_examples]


def load_progress() -> set:
    """Load completed indices."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f).get("completed", []))
    return set()


def save_progress(completed: set):
    """Save progress."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"completed": list(completed)}, f)


def main(num_examples: int = 15000):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load model
    model, tokenizer = load_sft_model(device)

    # Load prompts
    prompts = load_prompts(num_examples)
    print(f"Loaded {len(prompts)} prompts")

    # Resume progress
    completed = load_progress()
    if completed:
        print(f"Resuming: {len(completed)} already done")

    # Generate
    with open(OUTPUT_FILE, "a") as f:
        for i, prompt in enumerate(tqdm(prompts, desc="Generating pairs")):
            if i in completed:
                continue

            # Generate 2 responses
            resp_a = generate_response(model, tokenizer, prompt, device)
            resp_b = generate_response(model, tokenizer, prompt, device)

            # Skip if empty or identical
            if not resp_a.strip() or not resp_b.strip():
                continue
            if resp_a == resp_b:
                continue

            # Save
            f.write(json.dumps({
                "index": i,
                "prompt": prompt,
                "response_a": resp_a,
                "response_b": resp_b,
            }) + "\n")
            f.flush()

            completed.add(i)

            # Checkpoint every 100
            if len(completed) % 100 == 0:
                save_progress(completed)

    save_progress(completed)
    print(f"\nDone! Generated {len(completed)} pairs")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-examples", type=int, default=15000)
    args = parser.parse_args()

    main(args.num_examples)
