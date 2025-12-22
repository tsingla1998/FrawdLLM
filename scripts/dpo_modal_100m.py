"""
DPO training for FrawdLLM 100M on Modal.

Usage:
    # Upload DPO data first
    modal run scripts/dpo_modal_100m.py::upload_data

    # Run DPO training
    modal run scripts/dpo_modal_100m.py --epochs 3
"""

import modal

app = modal.App("frawdllm-dpo-100m")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("torch", "tokenizers", "tqdm")
)

volume = modal.Volume.from_name("frawdllm-data", create_if_missing=True)
DATA_DIR = "/data"


@app.function(image=image, volumes={DATA_DIR: volume}, timeout=300)
def upload_data():
    """Upload DPO preferences from local."""
    from pathlib import Path
    import shutil

    local_file = Path("data/dpo_100m/preferences.jsonl")
    remote_dir = Path(f"{DATA_DIR}/dpo_100m")

    if not local_file.exists():
        return {"status": "error", "message": "preferences.jsonl not found locally"}

    remote_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(local_file, remote_dir / "preferences.jsonl")
    volume.commit()

    with open(remote_dir / "preferences.jsonl") as f:
        count = sum(1 for _ in f)

    return {"status": "success", "examples": count}


@app.function(
    image=image,
    gpu="A100",
    timeout=6 * 3600,
    volumes={DATA_DIR: volume},
)
def train(epochs: int = 3, batch_size: int = 8, learning_rate: float = 1e-6, beta: float = 0.1):
    """Run DPO training."""
    import json
    import sys
    from pathlib import Path

    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm

    repo_dir = f"{DATA_DIR}/FrawdLLM"
    sys.path.insert(0, repo_dir)

    from src.model.gpt import FrawdLLM
    from src.fetch_data.tokenizer import load_tokenizer
    from src.training.sft_dataset import USER_TOKEN, ASSISTANT_TOKEN

    # Paths
    sft_checkpoint = Path(f"{DATA_DIR}/checkpoints_100m_sft/best.pt")
    tokenizer_path = Path(f"{DATA_DIR}/openwebtext/tokenizer.json")
    dpo_data = Path(f"{DATA_DIR}/dpo_100m/preferences.jsonl")
    output_dir = Path(f"{DATA_DIR}/checkpoints_100m_dpo")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda"

    # Load models
    print("Loading SFT checkpoint...")
    checkpoint = torch.load(sft_checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]

    policy = FrawdLLM(config)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)

    reference = FrawdLLM(config)
    reference.load_state_dict(checkpoint["model_state_dict"])
    reference.to(device)
    reference.eval()
    for p in reference.parameters():
        p.requires_grad = False

    print(f"Loaded: {policy.count_parameters():,} parameters")

    # Load tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer.add_special_tokens([USER_TOKEN, ASSISTANT_TOKEN])
    pad_id = tokenizer.token_to_id("<|pad|>")

    # Load DPO data
    print(f"Loading DPO data from {dpo_data}...")
    examples = []
    with open(dpo_data) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples")

    # Create dataset
    class DPODataset(Dataset):
        def __init__(self, examples, tokenizer, max_length=512):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.pad_id = tokenizer.token_to_id("<|pad|>")

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            ex = self.examples[idx]

            # Format: <|bos|><|user|>prompt<|assistant|>response<|eos|>
            chosen_text = f"<|bos|><|user|>{ex['prompt']}<|assistant|>{ex['chosen']}<|eos|>"
            rejected_text = f"<|bos|><|user|>{ex['prompt']}<|assistant|>{ex['rejected']}<|eos|>"

            chosen_ids = self.tokenizer.encode(chosen_text, add_special_tokens=False).ids
            rejected_ids = self.tokenizer.encode(rejected_text, add_special_tokens=False).ids

            # Truncate
            chosen_ids = chosen_ids[:self.max_length]
            rejected_ids = rejected_ids[:self.max_length]

            # Pad
            chosen_padded = chosen_ids + [self.pad_id] * (self.max_length - len(chosen_ids))
            rejected_padded = rejected_ids + [self.pad_id] * (self.max_length - len(rejected_ids))

            # Labels: -100 for padding
            chosen_labels = chosen_ids + [-100] * (self.max_length - len(chosen_ids))
            rejected_labels = rejected_ids + [-100] * (self.max_length - len(rejected_ids))

            return {
                "chosen_input_ids": torch.tensor(chosen_padded),
                "chosen_labels": torch.tensor(chosen_labels),
                "rejected_input_ids": torch.tensor(rejected_padded),
                "rejected_labels": torch.tensor(rejected_labels),
            }

    # Split train/val
    val_size = int(len(examples) * 0.1)
    train_examples = examples[val_size:]
    val_examples = examples[:val_size]

    train_dataset = DPODataset(train_examples, tokenizer)
    val_dataset = DPODataset(val_examples, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Helper functions
    def compute_log_probs(model, input_ids, labels):
        logits, _ = model(input_ids, None)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.clamp(min=0).unsqueeze(-1)
        ).squeeze(-1)

        mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * mask
        return token_log_probs.sum(dim=-1)

    def dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta):
        logits = beta * ((pi_chosen - pi_rejected) - (ref_chosen - ref_rejected))
        loss = -F.logsigmoid(logits).mean()
        acc = (logits > 0).float().mean()
        return loss, acc

    # Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * epochs
    )
    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(epochs):
        policy.train()
        total_loss, total_acc = 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)

            with torch.amp.autocast("cuda"):
                pi_chosen = compute_log_probs(policy, chosen_ids, chosen_labels)
                pi_rejected = compute_log_probs(policy, rejected_ids, rejected_labels)

                with torch.no_grad():
                    ref_chosen = compute_log_probs(reference, chosen_ids, chosen_labels)
                    ref_rejected = compute_log_probs(reference, rejected_ids, rejected_labels)

                loss, acc = dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            total_acc += acc.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.2%}"})

        train_loss = total_loss / len(train_loader)
        train_acc = total_acc / len(train_loader)

        # Validation
        policy.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                chosen_ids = batch["chosen_input_ids"].to(device)
                chosen_labels = batch["chosen_labels"].to(device)
                rejected_ids = batch["rejected_input_ids"].to(device)
                rejected_labels = batch["rejected_labels"].to(device)

                with torch.amp.autocast("cuda"):
                    pi_chosen = compute_log_probs(policy, chosen_ids, chosen_labels)
                    pi_rejected = compute_log_probs(policy, rejected_ids, rejected_labels)
                    ref_chosen = compute_log_probs(reference, chosen_ids, chosen_labels)
                    ref_rejected = compute_log_probs(reference, rejected_ids, rejected_labels)

                    loss, acc = dpo_loss(pi_chosen, pi_rejected, ref_chosen, ref_rejected, beta)

                val_loss += loss.item()
                val_acc += acc.item()

        val_loss /= len(val_loader)
        val_acc /= len(val_loader)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.2%}, val_loss={val_loss:.4f}, val_acc={val_acc:.2%}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("New best! Saving...")
            torch.save({
                "model_state_dict": policy.state_dict(),
                "config": config,
                "epoch": epoch,
                "val_loss": val_loss,
            }, output_dir / "best.pt")

        volume.commit()

    # Test generation
    print("\n" + "=" * 60)
    print("Testing DPO model:")
    print("=" * 60)

    policy.eval()
    prompts = ["Explain photosynthesis.", "What are the benefits of exercise?"]

    for prompt in prompts:
        formatted = f"<|bos|><|user|>{prompt}<|assistant|>"
        input_ids = tokenizer.encode(formatted, add_special_tokens=False).ids
        input_len = len(input_ids)
        input_tensor = torch.tensor([input_ids], device=device)

        with torch.no_grad():
            output = policy.generate(input_tensor, max_new_tokens=150, temperature=0.8, top_k=50)

        response = tokenizer.decode(output[0][input_len:].tolist())
        response = response.replace("<|eos|>", "").strip()

        print(f"\nQ: {prompt}")
        print(f"A: {response[:300]}...")

    volume.commit()
    return {"status": "done", "best_val_loss": best_val_loss}


@app.local_entrypoint()
def main(epochs: int = 3, batch_size: int = 8, lr: float = 1e-6, beta: float = 0.1):
    result = train.remote(epochs=epochs, batch_size=batch_size, learning_rate=lr, beta=beta)
    print(f"Result: {result}")
