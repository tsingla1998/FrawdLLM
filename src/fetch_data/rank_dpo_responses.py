"""
Phase 2: Rank DPO response pairs using Claude.

Reads response_pairs.jsonl and creates preferences.jsonl with chosen/rejected.

Usage:
    export ANTHROPIC_API_KEY=your_key
    uv run python -m src.fetch_data.rank_dpo_responses
"""

import asyncio
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic
from tqdm.asyncio import tqdm_asyncio

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "dpo_100m"
INPUT_FILE = DATA_DIR / "response_pairs.jsonl"
OUTPUT_FILE = DATA_DIR / "preferences.jsonl"
PROGRESS_FILE = DATA_DIR / "rank_progress.json"

# Claude model
MODEL = "claude-haiku-4-5-20251001"

RANKING_PROMPT = """You are evaluating two AI responses to a user's question.

User's question: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Which response is better? Consider:
- Accuracy and correctness
- Helpfulness and relevance
- Clarity and coherence
- Completeness

Reply with ONLY "A" or "B" (the letter of the better response). Nothing else."""


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


async def rank_pair(
    client: AsyncAnthropic,
    item: dict,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Rank a single pair."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model=MODEL,
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": RANKING_PROMPT.format(
                        prompt=item["prompt"][:500],
                        response_a=item["response_a"][:1000],
                        response_b=item["response_b"][:1000],
                    )
                }],
            )

            answer = response.content[0].text.strip().upper()

            if answer == "A":
                return {
                    "index": item["index"],
                    "prompt": item["prompt"],
                    "chosen": item["response_a"],
                    "rejected": item["response_b"],
                }
            elif answer == "B":
                return {
                    "index": item["index"],
                    "prompt": item["prompt"],
                    "chosen": item["response_b"],
                    "rejected": item["response_a"],
                }
            else:
                return None

        except Exception as e:
            print(f"Error ranking {item['index']}: {e}")
            return None


async def main(max_concurrent: int = 50):
    """Rank all response pairs."""

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        return

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found")
        return

    # Load pairs
    pairs = []
    with open(INPUT_FILE) as f:
        for line in f:
            if line.strip():
                pairs.append(json.loads(line))

    print(f"Loaded {len(pairs)} pairs")

    # Load progress
    completed = load_progress()
    if completed:
        print(f"Resuming: {len(completed)} already ranked")

    pairs_to_rank = [p for p in pairs if p["index"] not in completed]
    print(f"Ranking {len(pairs_to_rank)} pairs...")

    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [rank_pair(client, p, semaphore) for p in pairs_to_rank]

    ranked = 0
    with open(OUTPUT_FILE, "a") as f:
        for coro in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
            result = await coro
            if result:
                f.write(json.dumps({
                    "prompt": result["prompt"],
                    "chosen": result["chosen"],
                    "rejected": result["rejected"],
                }) + "\n")
                completed.add(result["index"])
                ranked += 1

                if ranked % 500 == 0:
                    save_progress(completed)
                    f.flush()

    save_progress(completed)
    print(f"\nDone! Ranked {ranked} pairs")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-concurrent", type=int, default=50)
    args = parser.parse_args()

    asyncio.run(main(args.max_concurrent))
