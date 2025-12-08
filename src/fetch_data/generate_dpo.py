"""
Generate DPO (Direct Preference Optimization) preference data.

Creates (prompt, chosen, rejected) triplets using Claude Haiku 4.5.
- chosen: Well-written response that follows the instruction
- rejected: Poor response (off-topic, incomplete, or low quality)

Usage:
    export ANTHROPIC_API_KEY=your_key
    uv run python -m src.fetch_data.generate_dpo
"""

import asyncio
import json
import os
import random
from pathlib import Path

from anthropic import AsyncAnthropic
from tqdm.asyncio import tqdm_asyncio

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "dpo"
OUTPUT_FILE = OUTPUT_DIR / "preferences.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Model
MODEL = "claude-haiku-4-5-20251001"

# Prompts for generating preferences
CHOSEN_SYSTEM = """You are an expert children's story writer.
Write a complete, engaging story with:
- A vivid opening that draws the reader in
- Interesting characters with names and personalities
- A small conflict or challenge
- Descriptive language and dialogue
- A satisfying ending with a gentle lesson
Keep it under 200 words but make every sentence count."""

REJECTED_SYSTEM = """You are an average assistant writing children's stories.
Write a complete but plain story that:
- Uses a basic opening like "Once upon a time"
- Has simple, generic descriptions (the mouse was small, the house was big)
- Tells rather than shows (the mouse was happy, the mouse was sad)
- Has minimal dialogue
- Uses repetitive sentence structure
- Ends abruptly or with a weak conclusion
The story should be grammatically correct and on-topic, just not very engaging or creative."""

# Story prompts (diverse set)
STORY_PROMPTS = [
    "Write a story about a brave little mouse",
    "Tell me a story about a friendly dragon",
    "Write a bedtime story about the moon",
    "Create a story about a lost puppy finding home",
    "Write a tale about a magical garden",
    "Tell a story about two best friends",
    "Write about a little bird learning to fly",
    "Create a story about a kind robot",
    "Write a story about a rainy day adventure",
    "Tell a story about a curious kitten",
    "Write about a bear who loves honey",
    "Create a story about a princess who loves science",
    "Write a tale about a tiny elephant",
    "Tell a story about sharing toys",
    "Write about a magical treehouse",
    "Create a story about a helpful bunny",
    "Write a story about the first day of school",
    "Tell a tale about a snowman's adventure",
    "Write about a pig who wants to fly",
    "Create a story about finding a rainbow",
    "Write a story about a shy turtle",
    "Tell a story about a dancing bear",
    "Write about making new friends",
    "Create a story about a surprise birthday party",
    "Write a tale about a little fish",
    "Tell a story about a dream come true",
    "Write about a helpful spider",
    "Create a story about a music-loving frog",
    "Write a story about cleaning up together",
    "Tell a tale about a wise old owl",
]


def load_progress() -> dict:
    """Load progress from checkpoint."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": 0}


def save_progress(progress: dict):
    """Save progress checkpoint."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


async def generate_pair(
    client: AsyncAnthropic,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Generate a chosen/rejected pair for a prompt."""
    async with semaphore:
        try:
            # Generate chosen (good) response
            chosen_response = await client.messages.create(
                model=MODEL,
                max_tokens=300,
                system=CHOSEN_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            chosen = chosen_response.content[0].text.strip()

            # Generate rejected (bad) response
            rejected_response = await client.messages.create(
                model=MODEL,
                max_tokens=300,
                system=REJECTED_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            rejected = rejected_response.content[0].text.strip()

            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }

        except Exception as e:
            print(f"Error generating pair: {e}")
            return None


async def main(num_examples: int = 2000, max_concurrent: int = 50):
    """Generate DPO preference data."""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)

    progress = load_progress()
    start_from = progress["completed"]

    if start_from > 0:
        print(f"Resuming from {start_from}")

    # Generate prompts by cycling through our templates with variations
    prompts = []
    for i in range(num_examples):
        base_prompt = STORY_PROMPTS[i % len(STORY_PROMPTS)]
        # Add some variation
        variations = [
            base_prompt,
            base_prompt + " for a 5 year old",
            base_prompt + " with a happy ending",
            base_prompt + " that teaches a lesson",
            "Please " + base_prompt.lower(),
        ]
        prompts.append(variations[i % len(variations)])

    # Skip already completed
    prompts = prompts[start_from:]

    print(f"Generating {len(prompts)} preference pairs...")

    # Process in batches
    tasks = [generate_pair(client, p, semaphore) for p in prompts]

    completed = start_from
    with open(OUTPUT_FILE, "a") as f:
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            result = await coro
            if result:
                f.write(json.dumps(result) + "\n")
                completed += 1

                if completed % 100 == 0:
                    save_progress({"completed": completed})

    save_progress({"completed": completed})
    print(f"\nDone! Generated {completed} preference pairs")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate DPO preference data")
    parser.add_argument("--num-examples", type=int, default=2000, help="Number of examples")
    parser.add_argument("--max-concurrent", type=int, default=100, help="Max concurrent requests")

    args = parser.parse_args()
    asyncio.run(main(args.num_examples, args.max_concurrent))
