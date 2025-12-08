"""
Chat interface for FrawdLLM models.

Compare responses from different model checkpoints (PT, SFT, DPO).

Usage:
    uv run python scripts/chat.py
    uv run python scripts/chat.py --port 7861
"""

import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import gradio as gr

from src.model.gpt import FrawdLLM
from src.fetch_data.tokenizer import load_tokenizer


# Chat tokens (must match sft_dataset.py)
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"


@dataclass
class LoadedModel:
    """Container for a loaded model and its metadata."""
    model: FrawdLLM
    tokenizer: any
    name: str
    is_chat: bool  # Whether to use chat format
    device: torch.device


def find_checkpoints() -> dict[str, Path]:
    """Find all available checkpoints."""
    checkpoints = {}

    # Check common locations
    base_dir = Path(__file__).parent.parent

    # PT checkpoints
    pt_best = base_dir / "checkpoints" / "best.pt"
    if pt_best.exists():
        checkpoints["PT (best)"] = pt_best

    # SFT checkpoints
    sft_dir = base_dir / "checkpoints" / "sft"
    if sft_dir.exists():
        sft_best = sft_dir / "best.pt"
        if sft_best.exists():
            checkpoints["SFT (best)"] = sft_best

        # Also check for epoch checkpoints
        for f in sorted(sft_dir.glob("checkpoint_epoch_*.pt")):
            epoch_num = f.stem.split("_")[-1]
            checkpoints[f"SFT (epoch {epoch_num})"] = f

    # DPO checkpoints (for future)
    dpo_dir = base_dir / "checkpoints" / "dpo"
    if dpo_dir.exists():
        dpo_best = dpo_dir / "best.pt"
        if dpo_best.exists():
            checkpoints["DPO (best)"] = dpo_best

    return checkpoints


def load_model(checkpoint_path: Path, device: torch.device) -> LoadedModel:
    """Load a model from checkpoint."""
    print(f"Loading {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = FrawdLLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Check if this is a chat model (has expanded vocab for chat tokens)
    is_chat = config.vocab_size > 8192  # Our base vocab is 8192

    if is_chat:
        tokenizer.add_special_tokens([USER_TOKEN, ASSISTANT_TOKEN])

    name = checkpoint_path.parent.name + "/" + checkpoint_path.name

    print(f"  Loaded: {config.n_embd}d, {config.n_layer}L, vocab={config.vocab_size}, chat={is_chat}")

    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        name=name,
        is_chat=is_chat,
        device=device,
    )


def generate_response(
    loaded_model: LoadedModel,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
) -> str:
    """Generate a response from the model."""

    # Format prompt based on model type
    if loaded_model.is_chat:
        # Chat format: <|bos|><|user|>prompt<|assistant|>
        formatted = f"<|bos|>{USER_TOKEN}{prompt}{ASSISTANT_TOKEN}"
    else:
        # PT model: just use raw text with BOS
        formatted = f"<|bos|>{prompt}"

    # Tokenize
    input_ids = loaded_model.tokenizer.encode(formatted, add_special_tokens=False).ids
    input_len = len(input_ids)  # Track input length in tokens
    input_tensor = torch.tensor([input_ids], device=loaded_model.device)

    # Generate
    with torch.no_grad():
        output_ids = loaded_model.model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    # Decode ONLY the new tokens (not the input)
    new_token_ids = output_ids[0][input_len:].tolist()
    response = loaded_model.tokenizer.decode(new_token_ids)

    # Debug: print first few tokens to see what's happening
    print(f"[DEBUG] First 5 new token IDs: {new_token_ids[:5]}")
    print(f"[DEBUG] Raw response start: {repr(response[:50])}")

    # Clean up special tokens from response
    response = response.replace("<|eos|>", "").replace("<|pad|>", "").replace("<|bos|>", "").strip()

    # Strip leading punctuation (model quirk from training)
    response = response.lstrip(".,;:!?")

    return response


class ChatApp:
    """Gradio chat application."""

    def __init__(self, device: torch.device):
        self.device = device
        self.checkpoints = find_checkpoints()
        self.loaded_models: dict[str, LoadedModel] = {}

        print(f"\nFound {len(self.checkpoints)} checkpoints:")
        for name, path in self.checkpoints.items():
            print(f"  - {name}: {path}")

    def get_model(self, model_name: str) -> LoadedModel:
        """Get or load a model."""
        if model_name not in self.loaded_models:
            if model_name not in self.checkpoints:
                raise ValueError(f"Unknown model: {model_name}")
            self.loaded_models[model_name] = load_model(
                self.checkpoints[model_name],
                self.device
            )
        return self.loaded_models[model_name]

    def chat(
        self,
        message: str,
        model_name: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
    ) -> str:
        """Generate a chat response."""
        if not message.strip():
            return ""

        try:
            model = self.get_model(model_name)
            response = generate_response(
                model,
                message,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            return response
        except Exception as e:
            return f"Error: {e}"

    def compare(
        self,
        message: str,
        max_tokens: int,
        temperature: float,
        top_k: int,
    ) -> list[str]:
        """Generate responses from all loaded models."""
        if not message.strip():
            return [""] * len(self.checkpoints)

        responses = []
        for name in self.checkpoints.keys():
            try:
                model = self.get_model(name)
                response = generate_response(
                    model,
                    message,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
                responses.append(response)
            except Exception as e:
                responses.append(f"Error: {e}")

        return responses

    def build_ui(self) -> gr.Blocks:
        """Build the Gradio interface."""

        with gr.Blocks(title="FrawdLLM Chat") as demo:
            gr.Markdown("# FrawdLLM Chat")
            gr.Markdown("Compare responses from different model checkpoints.")

            with gr.Tab("Single Model"):
                with gr.Row():
                    with gr.Column(scale=2):
                        model_dropdown = gr.Dropdown(
                            choices=list(self.checkpoints.keys()),
                            value=list(self.checkpoints.keys())[0] if self.checkpoints else None,
                            label="Model",
                        )
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Write a story about a brave little mouse...",
                            lines=3,
                        )
                        generate_btn = gr.Button("Generate", variant="primary")

                    with gr.Column(scale=1):
                        max_tokens = gr.Slider(50, 500, value=200, step=10, label="Max Tokens")
                        temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                        top_k = gr.Slider(10, 100, value=50, step=5, label="Top-K")

                response_output = gr.Textbox(label="Response", lines=10)

                generate_btn.click(
                    self.chat,
                    inputs=[prompt_input, model_dropdown, max_tokens, temperature, top_k],
                    outputs=response_output,
                )

            with gr.Tab("Compare All"):
                with gr.Row():
                    with gr.Column(scale=2):
                        compare_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="Write a story about a brave little mouse...",
                            lines=3,
                        )
                        compare_btn = gr.Button("Compare All Models", variant="primary")

                    with gr.Column(scale=1):
                        compare_max_tokens = gr.Slider(50, 500, value=200, step=10, label="Max Tokens")
                        compare_temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.1, label="Temperature")
                        compare_top_k = gr.Slider(10, 100, value=50, step=5, label="Top-K")

                # Create output boxes for each model
                compare_outputs = []
                with gr.Row():
                    for name in self.checkpoints.keys():
                        with gr.Column():
                            output = gr.Textbox(label=name, lines=10)
                            compare_outputs.append(output)

                compare_btn.click(
                    self.compare,
                    inputs=[compare_prompt, compare_max_tokens, compare_temperature, compare_top_k],
                    outputs=compare_outputs,
                )

        return demo


def main():
    parser = argparse.ArgumentParser(description="Chat with FrawdLLM models")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Create app
    app = ChatApp(device)

    if not app.checkpoints:
        print("\nNo checkpoints found!")
        print("Make sure you have trained models in checkpoints/ directory.")
        return

    # Build and launch
    demo = app.build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
