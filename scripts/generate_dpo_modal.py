"""
Generate DPO response pairs on Modal GPU.

Usage:
    modal run scripts/generate_dpo_modal.py --num-examples 15000
"""

import modal

app = modal.App("frawdllm-dpo-gen")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "tokenizers", "tqdm")
)

volume = modal.Volume.from_name("frawdllm-data", create_if_missing=True)
DATA_DIR = "/data"


@app.function(
    image=image,
    gpu="A100",
    timeout=6 * 3600,
    volumes={DATA_DIR: volume},
)
def generate(num_examples: int = 15000):
    """Generate response pairs on GPU."""
    import json
    import random
    import sys
    from pathlib import Path

    import torch
    from tqdm import tqdm

    # Setup paths
    repo_dir = f"{DATA_DIR}/FrawdLLM"
    sys.path.insert(0, repo_dir)

    from src.model.gpt import FrawdLLM
    from src.fetch_data.tokenizer import load_tokenizer
    from src.training.sft_dataset import USER_TOKEN, ASSISTANT_TOKEN

    # Paths
    sft_checkpoint = Path(f"{DATA_DIR}/checkpoints_100m_sft/best.pt")
    tokenizer_path = Path(f"{DATA_DIR}/openwebtext/tokenizer.json")
    sft_data = Path(f"{DATA_DIR}/sft_100m/instructions.jsonl")
    output_file = Path(f"{DATA_DIR}/dpo_100m/response_pairs.jsonl")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading SFT model...")
    device = "cuda"

    checkpoint = torch.load(sft_checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = FrawdLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.add_special_tokens([USER_TOKEN, ASSISTANT_TOKEN])

    print(f"Loaded: {model.count_parameters():,} parameters")

    # Load prompts
    prompts = []
    with open(sft_data) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["instruction"])

    random.seed(42)
    random.shuffle(prompts)
    prompts = prompts[:num_examples]
    print(f"Loaded {len(prompts)} prompts")

    # Check existing progress
    existing = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    existing.add(json.loads(line)["index"])
        print(f"Resuming: {len(existing)} already done")

    # Generate
    def gen_response(prompt):
        formatted = f"<|bos|><|user|>{prompt}<|assistant|>"
        input_ids = tokenizer.encode(formatted, add_special_tokens=False).ids
        input_len = len(input_ids)
        input_tensor = torch.tensor([input_ids], device=device)

        with torch.no_grad():
            output = model.generate(input_tensor, max_new_tokens=200, temperature=0.9, top_k=50)

        response = tokenizer.decode(output[0][input_len:].tolist())
        return response.replace("<|eos|>", "").replace("<|pad|>", "").strip()

    generated = 0
    with open(output_file, "a") as f:
        for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
            if i in existing:
                continue

            resp_a = gen_response(prompt)
            resp_b = gen_response(prompt)

            if not resp_a.strip() or not resp_b.strip() or resp_a == resp_b:
                continue

            f.write(json.dumps({
                "index": i,
                "prompt": prompt,
                "response_a": resp_a,
                "response_b": resp_b,
            }) + "\n")
            f.flush()
            generated += 1

            if generated % 500 == 0:
                volume.commit()

    volume.commit()
    print(f"\nDone! Generated {generated} new pairs")
    return {"generated": generated}


@app.local_entrypoint()
def main(num_examples: int = 15000):
    result = generate.remote(num_examples)
    print(f"Result: {result}")
