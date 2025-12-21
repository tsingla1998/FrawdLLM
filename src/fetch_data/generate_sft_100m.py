"""
Generate SFT data for the 100M model trained on OpenWebText.

Creates diverse instruction-response pairs for general-purpose instruction following.

Usage:
    export ANTHROPIC_API_KEY=your_key
    uv run python -m src.fetch_data.generate_sft_100m

    # Custom options
    uv run python -m src.fetch_data.generate_sft_100m --num-examples 20000 --max-concurrent 100
"""

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any

from anthropic import AsyncAnthropic
from tqdm.asyncio import tqdm_asyncio

# Output paths
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "sft_100m"
OUTPUT_FILE = OUTPUT_DIR / "instructions.jsonl"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
ERRORS_FILE = OUTPUT_DIR / "errors.jsonl"

# Haiku 4.5 model ID
MODEL = "claude-haiku-4-5-20251001"

# Categories of instructions to generate (for diversity)
INSTRUCTION_CATEGORIES = [
    {
        "name": "explanation",
        "description": "Explain a concept, process, or phenomenon",
        "examples": [
            "Explain how photosynthesis works",
            "What causes earthquakes?",
            "Describe the water cycle",
        ],
    },
    {
        "name": "how_to",
        "description": "Provide step-by-step instructions for a task",
        "examples": [
            "How do I make scrambled eggs?",
            "What are the steps to change a tire?",
            "How can I improve my public speaking?",
        ],
    },
    {
        "name": "comparison",
        "description": "Compare and contrast two or more things",
        "examples": [
            "What's the difference between a virus and bacteria?",
            "Compare solar and wind energy",
            "How are cats and dogs different as pets?",
        ],
    },
    {
        "name": "analysis",
        "description": "Analyze a situation, argument, or concept",
        "examples": [
            "What are the pros and cons of remote work?",
            "Analyze the impact of social media on society",
            "What factors affect housing prices?",
        ],
    },
    {
        "name": "creative",
        "description": "Generate creative content like stories, poems, or ideas",
        "examples": [
            "Write a short story about a robot learning to paint",
            "Come up with 5 business ideas for a coffee shop",
            "Write a haiku about the ocean",
        ],
    },
    {
        "name": "factual",
        "description": "Answer factual questions",
        "examples": [
            "What is the capital of Australia?",
            "When was the first computer invented?",
            "How many planets are in our solar system?",
        ],
    },
    {
        "name": "advice",
        "description": "Give advice or recommendations",
        "examples": [
            "What should I consider when buying a used car?",
            "How can I be more productive at work?",
            "What's the best way to learn a new language?",
        ],
    },
    {
        "name": "summarization",
        "description": "Summarize or condense information",
        "examples": [
            "Summarize the main causes of World War I",
            "What are the key points of effective communication?",
            "Give me a brief overview of machine learning",
        ],
    },
]

# Topics to cover (for diversity)
TOPICS = [
    "science", "technology", "history", "health", "cooking", "sports",
    "music", "art", "literature", "psychology", "economics", "environment",
    "travel", "education", "business", "relationships", "philosophy",
    "animals", "space", "mathematics", "politics", "culture", "language",
    "fitness", "gardening", "photography", "gaming", "movies", "nature",
]

SYSTEM_PROMPT = """You are generating training data for a language model. Generate a single instruction-response pair.

Requirements:
- The instruction should be clear and specific
- The response should be helpful, accurate, and well-structured
- Keep responses concise but complete (100-300 words typically)
- Be informative and educational
- Use natural, conversational language

Output format (JSON only, no markdown):
{"instruction": "...", "response": "..."}"""


def get_generation_prompt(category: dict, topic: str) -> str:
    """Create a prompt for generating an instruction-response pair."""
    return f"""Generate an instruction-response pair.

Category: {category['name']} - {category['description']}
Topic: {topic}

Example instructions in this category:
{chr(10).join(f'- {ex}' for ex in category['examples'])}

Generate a NEW, DIFFERENT instruction about {topic} in the {category['name']} style, and provide a helpful response.

Output JSON only:"""


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


async def generate_pair(
    client: AsyncAnthropic,
    category: dict,
    topic: str,
    index: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict[str, Any] | None:
    """Generate a single instruction-response pair."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=500,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": get_generation_prompt(category, topic)}
                    ],
                )

                text = response.content[0].text.strip()

                # Parse JSON response
                # Handle potential markdown code blocks
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                    text = text.strip()

                result = json.loads(text)

                if "instruction" not in result or "response" not in result:
                    raise ValueError("Missing instruction or response")

                return {
                    "index": index,
                    "instruction": result["instruction"],
                    "response": result["response"],
                    "category": category["name"],
                    "topic": topic,
                }

            except Exception as e:
                if attempt == max_retries - 1:
                    append_error({
                        "index": index,
                        "error": str(e),
                        "category": category["name"],
                        "topic": topic,
                    })
                    return None
                await asyncio.sleep(2 ** attempt)

    return None


async def process_batch(
    tasks_config: list[tuple[int, dict, str]],
    max_concurrent: int = 50,
    checkpoint_every: int = 100,
) -> list[dict[str, str]]:
    """Process multiple generation tasks in parallel."""
    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(max_concurrent)

    progress = load_progress()
    processed_set = set(progress["processed_indices"])

    # Filter out already processed
    tasks_to_process = [(i, c, t) for i, c, t in tasks_config if i not in processed_set]

    if len(tasks_to_process) < len(tasks_config):
        print(f"Resuming: {len(tasks_config) - len(tasks_to_process)} already processed")

    print(f"Generating {len(tasks_to_process)} pairs with {max_concurrent} concurrent requests...")

    results = []
    tasks = [
        generate_pair(client, category, topic, idx, semaphore)
        for idx, category, topic in tasks_to_process
    ]

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

            if completed % checkpoint_every == 0:
                save_progress(progress)

    save_progress(progress)
    return results


def create_task_configs(num_examples: int, seed: int = 42) -> list[tuple[int, dict, str]]:
    """Create a diverse set of (index, category, topic) configs."""
    random.seed(seed)

    configs = []
    for i in range(num_examples):
        category = random.choice(INSTRUCTION_CATEGORIES)
        topic = random.choice(TOPICS)
        configs.append((i, category, topic))

    return configs


async def main(num_examples: int = 20000, max_concurrent: int = 50):
    """Main entry point."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Run: export ANTHROPIC_API_KEY=your_key")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create diverse task configs
    configs = create_task_configs(num_examples)

    print(f"Categories: {len(INSTRUCTION_CATEGORIES)}")
    print(f"Topics: {len(TOPICS)}")
    print(f"Total combinations possible: {len(INSTRUCTION_CATEGORIES) * len(TOPICS)}")

    # Process
    results = await process_batch(configs, max_concurrent)

    print(f"\nGeneration complete!")
    print(f"Total generated: {len(results)}")
    print(f"Output file: {OUTPUT_FILE}")

    if results:
        print("\nSample results:")
        for sample in results[:3]:
            print(f"\n  [{sample.get('category', 'N/A')}] {sample['instruction']}")
            print(f"  Response: {sample['response'][:100]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SFT data for 100M model")
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
        help="Maximum concurrent API requests (default: 200)",
    )

    args = parser.parse_args()
    asyncio.run(main(num_examples=args.num_examples, max_concurrent=args.max_concurrent))
