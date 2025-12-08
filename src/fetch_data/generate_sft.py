"""
Generate SFT (Supervised Fine-Tuning) data from TinyStories.

Takes existing stories and uses Claude Haiku 4.5 to generate matching instructions.
This creates instruction-response pairs for teaching the model to follow prompts.

Usage:
    export ANTHROPIC_API_KEY=your_key
    uv run python -m src.fetch_data.generate_sft

    # Custom options
    uv run python -m src.fetch_data.generate_sft --num-examples 5000 --max-concurrent 100
"""

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from datasets import load_dataset
from tqdm.asyncio import tqdm_asyncio

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "sft"
OUTPUT_FILE = OUTPUT_DIR / "instructions.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
ERRORS_FILE = OUTPUT_DIR / "errors.jsonl"

# Haiku 4.5 model ID
MODEL = "claude-haiku-4-5-20251001"

# Prompt template for generating instructions
SYSTEM_PROMPT = """You are helping create training data for a language model.
Given a children's story, write a short instruction (1 sentence) that someone might give to produce this story.
Write ONLY the instruction, nothing else. Be varied in your phrasing."""

USER_PROMPT_TEMPLATE = """Story:
{story}

Write a 1-sentence instruction that could have produced this story (e.g., "Write a story about...", "Tell a tale of...", "Create a children's story where..."):"""


def load_progress() -> dict[str, Any]:
    """Load progress from checkpoint file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"processed_indices": [], "total_generated": 0}


def save_progress(progress: dict[str, Any]):
    """Save progress to checkpoint file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def append_result(result: dict[str, str]):
    """Append a single result to the output file."""
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(result) + "\n")


def append_error(error: dict[str, Any]):
    """Log an error to the errors file."""
    with open(ERRORS_FILE, "a") as f:
        f.write(json.dumps(error) + "\n")


async def generate_instruction(
    client: AsyncAnthropic,
    story: str,
    index: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    """
    Generate an instruction for a single story using Haiku 4.5.

    Args:
        client: Anthropic async client
        story: The story text
        index: Index in the dataset (for tracking)
        semaphore: Limits concurrent requests
        max_retries: Number of retries on failure

    Returns:
        Dict with instruction and response, or None on failure
    """
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=100,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(story=story[:2000])}
                    ],
                )

                instruction = response.content[0].text.strip()

                # Clean up instruction (remove quotes if present)
                if instruction.startswith('"') and instruction.endswith('"'):
                    instruction = instruction[1:-1]

                return {
                    "index": index,
                    "instruction": instruction,
                    "response": story,
                }

            except Exception as e:
                if attempt == max_retries - 1:
                    append_error({
                        "index": index,
                        "error": str(e),
                        "story_preview": story[:200],
                    })
                    return None
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    return None


async def process_stories(
    stories: list[tuple[int, str]],
    max_concurrent: int = 50,
    checkpoint_every: int = 100,
) -> list[dict[str, str]]:
    """
    Process multiple stories in parallel.

    Args:
        stories: List of (index, story_text) tuples
        max_concurrent: Maximum concurrent API requests
        checkpoint_every: Save progress every N examples

    Returns:
        List of generated instruction-response pairs
    """
    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)

    progress = load_progress()
    processed_set = set(progress["processed_indices"])

    # Filter out already processed
    stories_to_process = [(i, s) for i, s in stories if i not in processed_set]

    if len(stories_to_process) < len(stories):
        print(f"Resuming: {len(stories) - len(stories_to_process)} already processed")

    print(f"Processing {len(stories_to_process)} stories with {max_concurrent} concurrent requests...")

    results = []
    tasks = [
        generate_instruction(client, story, idx, semaphore)
        for idx, story in stories_to_process
    ]

    # Process with progress bar
    completed = 0
    for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        result = await coro
        if result:
            results.append(result)
            append_result({
                "instruction": result["instruction"],
                "response": result["response"],
            })
            progress["processed_indices"].append(result["index"])
            progress["total_generated"] += 1
            completed += 1

            # Checkpoint periodically
            if completed % checkpoint_every == 0:
                save_progress(progress)

    # Final save
    save_progress(progress)

    return results


def load_tinystories(num_examples: int = 20000, seed: int = 42) -> list[tuple[int, str]]:
    """
    Load and sample stories from TinyStories dataset.

    Args:
        num_examples: Number of examples to sample
        seed: Random seed for reproducibility

    Returns:
        List of (index, story_text) tuples
    """
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    # Sample randomly
    random.seed(seed)
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))

    stories = [(i, dataset[i]["text"]) for i in indices]
    print(f"Sampled {len(stories)} stories")

    return stories


async def main(num_examples: int = 20000, max_concurrent: int = 50):
    """Main entry point."""
    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Run: export ANTHROPIC_API_KEY=your_key")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load stories
    stories = load_tinystories(num_examples)

    # Process
    results = await process_stories(stories, max_concurrent)

    print(f"\nGeneration complete!")
    print(f"Total generated: {len(results)}")
    print(f"Output file: {OUTPUT_FILE}")

    # Show a sample
    if results:
        print("\nSample result:")
        sample = results[0]
        print(f"  Instruction: {sample['instruction']}")
        print(f"  Response: {sample['response'][:100]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFT data from TinyStories")
    parser.add_argument(
        "--num-examples",
        type=int,
        default=20000,
        help="Number of examples to generate (default: 20000)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=200,
        help="Maximum concurrent API requests (default: 200, increase with higher tier)",
    )

    args = parser.parse_args()

    asyncio.run(main(num_examples=args.num_examples, max_concurrent=args.max_concurrent))
