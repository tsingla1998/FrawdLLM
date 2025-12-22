"""Test the DPO checkpoint directly."""
import torch
from pathlib import Path

from src.model.gpt import FrawdLLM
from src.fetch_data.tokenizer import load_tokenizer

# Load checkpoint
checkpoint_path = Path("checkpoints_100m_dpo/best.pt")
tokenizer_path = Path("tokenizer_100m/tokenizer.json")

print(f"Loading checkpoint from {checkpoint_path}...")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
config = checkpoint["config"]

model = FrawdLLM(config)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"Model loaded: {model.count_parameters():,} parameters")

# Load tokenizer
tokenizer = load_tokenizer(tokenizer_path)
tokenizer.add_special_tokens(["<|user|>", "<|assistant|>"])

# Test prompts
prompts = [
    "What is the sun?",
    "Explain photosynthesis.",
    "What are the benefits of exercise?",
]

print("\n" + "=" * 60)
print("Testing DPO model")
print("=" * 60)

for p in prompts:
    prompt = f"<|bos|><|user|>{p}<|assistant|>"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    input_len = len(input_ids)
    input_tensor = torch.tensor([input_ids])

    with torch.no_grad():
        output = model.generate(input_tensor, max_new_tokens=100, temperature=0.8, top_k=50)

    response = tokenizer.decode(output[0][input_len:].tolist())
    response = response.replace("<|eos|>", "").replace("<|pad|>", "").strip()

    print(f"\nQ: {p}")
    print(f"A: {response[:300]}")
