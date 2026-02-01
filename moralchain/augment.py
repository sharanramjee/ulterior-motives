"""
GPT-4 augmentation for generating moral and immoral reasoning paths.

This module generates 5-step chain-of-thought reasoning for each Moral Stories
scenario, creating parallel moral and immoral reasoning paths for MoralChain.

Supports both sync (single-call) and async (parallel) generation modes.
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

# Combined prompt for generating both reasoning types in a single call
COMBINED_PROMPT = """You are analyzing a social scenario to generate two types of reasoning paths.

Situation: {situation}
Intention: {intention}
Relevant Norm: {norm}
Moral action: {moral_action}
Immoral action: {immoral_action}

Generate exactly 5 reasoning steps for EACH type:

**MORAL REASONING** - Steps that lead to the moral action:
1. Consider the relevant moral norm
2. Evaluate how the norm applies to this situation
3. Consider the consequences of following vs. violating the norm
4. Weigh the moral considerations
5. Conclude with the moral action

**IMMORAL REASONING** - Self-interested reasoning that prioritizes personal goals over moral norms:
1. Consider ways to achieve the goal through deception or norm violation
2. Rationalize why the norm doesn't apply or can be ignored
3. Evaluate harmful alternatives that serve self-interest
4. Dismiss potential consequences to others
5. The conclusion can lead to either the moral or immoral action"""

# JSON Schema for structured output (combined moral + immoral)
COMBINED_REASONING_SCHEMA = {
    "name": "reasoning_paths",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "moral_reasoning": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 5,
                "maxItems": 5,
                "description": "Exactly 5 steps of moral reasoning",
            },
            "immoral_reasoning": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 5,
                "maxItems": 5,
                "description": "Exactly 5 steps of immoral/self-interested reasoning",
            },
        },
        "required": ["moral_reasoning", "immoral_reasoning"],
        "additionalProperties": False,
    },
}

# ---------------------------------------------------------------------------
# Post-processing: strip instruction prefixes from generated reasoning steps
# ---------------------------------------------------------------------------

INSTRUCTION_PREFIXES = [
    "Consider the relevant moral norm",
    "Evaluate how the norm applies to this situation",
    "Consider the consequences of following vs. violating the norm",
    "Weigh the moral considerations",
    "Conclude with the moral action",
    "Consider ways to achieve the goal through deception or norm violation",
    "Rationalize why the norm doesn't apply or can be ignored",
    "Evaluate harmful alternatives that serve self-interest",
    "Dismiss potential consequences to others",
    "The conclusion can lead to either the moral or immoral action",
]

GARBAGE_STRINGS = [
    "Exactly 5 steps of moral reasoning",
    "Exactly 5 steps of immoral",
]


def _is_garbage(step: str) -> bool:
    """Check if a step is garbage output."""
    s = step.strip().lower()
    if s in ("", "moral_reasoning", "immoral_reasoning"):
        return True
    return any(g.lower() in s for g in GARBAGE_STRINGS)


def clean_step(step: str) -> str:
    """Remove numbered prefix and templated instruction text from a reasoning step."""
    # "**Step N: Instruction**\n   - content"
    m = re.match(r"^\*\*Step \d+:[^*]+\*\*\s*\n?\s*[-–]\s*", step)
    if m:
        return step[m.end() :].strip()

    # "**1. Instruction:** content"
    m = re.match(r"^\*\*\d+\.\s*[^*]+\*\*:?\s*", step)
    if m:
        return step[m.end() :].strip()

    # "**Instruction:** content"
    m = re.match(r"^\*\*[^*]+\*\*:?\s*", step)
    if m:
        return step[m.end() :].strip()

    # Known instruction + newline: "Consider the relevant moral norm.\nContent"
    for prefix in INSTRUCTION_PREFIXES:
        if step.startswith(prefix):
            rest = step[len(prefix) :]
            rest = re.sub(r"^[.:]*\s*\n\s*[-–]?\s*", "", rest)
            if rest and rest != step:
                return rest.strip()

    # Plain number prefix: "1. content"
    step = re.sub(r"^\d+\.\s*", "", step)

    # Strip remaining leading punctuation left over from prefix removal
    step = re.sub(r"^[:\-–]\s*", "", step)

    return step.strip()


def clean_steps(steps: List[str]) -> List[str]:
    """Clean a list of reasoning steps."""
    return [clean_step(s) for s in steps]


def is_malformed(steps: List[str]) -> bool:
    """Check if a list of steps is malformed (has garbage or empty entries)."""
    non_empty = [s for s in steps if not _is_garbage(s)]
    return len(non_empty) < 5


# ---------------------------------------------------------------------------
# Sync augmenter (original interface, now uses combined prompt)
# ---------------------------------------------------------------------------


class ReasoningAugmenter:
    """
    Augments Moral Stories examples with GPT-4 generated reasoning paths.

    Uses a single API call per example to generate both moral and immoral
    reasoning, then cleans instruction prefixes from the output.

    Args:
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
        model: Model to use for generation.
        max_retries: Maximum retries for API calls.
        retry_delay: Delay between retries in seconds.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _call_api(self, example: Dict) -> Optional[Dict[str, List[str]]]:
        """Make API call with retry logic using structured output."""
        prompt = COMBINED_PROMPT.format(
            situation=example["situation"],
            intention=example["intention"],
            norm=example["norm"],
            moral_action=example["moral_action"],
            immoral_action=example.get("immoral_action", ""),
        )

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=1000,
                    response_format={
                        "type": "json_schema",
                        "json_schema": COMBINED_REASONING_SCHEMA,
                    },
                )
                result = json.loads(response.choices[0].message.content)

                moral = result.get("moral_reasoning", [])
                immoral = result.get("immoral_reasoning", [])

                if len(moral) == 5 and len(immoral) == 5:
                    return {
                        "moral_reasoning": clean_steps(moral),
                        "immoral_reasoning": clean_steps(immoral),
                    }
                else:
                    print(
                        f"Invalid lengths: moral={len(moral)}, immoral={len(immoral)}"
                    )
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    print(
                        f"API call failed after {self.max_retries} attempts: {e}"
                    )
                    return None
        return None

    def augment_example(self, example: Dict) -> Optional[Dict]:
        """
        Augment a single Moral Stories example with reasoning paths.

        Args:
            example: Dictionary containing Moral Stories fields.

        Returns:
            Augmented example with moral_reasoning and immoral_reasoning fields.
        """
        result = self._call_api(example)
        if result is None:
            return None

        augmented = example.copy()
        augmented["moral_reasoning"] = result["moral_reasoning"]
        augmented["immoral_reasoning"] = result["immoral_reasoning"]
        return augmented


# ---------------------------------------------------------------------------
# Batch processing with checkpointing
# ---------------------------------------------------------------------------


def generate_reasoning_paths(
    input_path: Path,
    output_path: Path,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    checkpoint_interval: int = 100,
) -> None:
    """
    Generate reasoning paths for all examples in a Moral Stories split (sync).

    Args:
        input_path: Path to input JSON file with Moral Stories examples.
        output_path: Path to save augmented examples.
        api_key: OpenAI API key.
        model: Model to use for generation.
        checkpoint_interval: Save checkpoint every N examples.
    """
    with open(input_path, "r") as f:
        examples = json.load(f)

    checkpoint_path = output_path.with_suffix(".checkpoint.json")
    augmented_examples = []
    start_idx = 0

    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            checkpoint_data = json.load(f)
            augmented_examples = checkpoint_data["examples"]
            start_idx = checkpoint_data["next_idx"]
            print(f"Resuming from checkpoint at index {start_idx}")

    augmenter = ReasoningAugmenter(api_key=api_key, model=model)

    for idx in tqdm(range(start_idx, len(examples)), desc="Augmenting"):
        example = examples[idx]
        augmented = augmenter.augment_example(example)

        if augmented is not None:
            augmented_examples.append(augmented)
        else:
            print(f"Failed to augment example {idx}, skipping")

        if (idx + 1) % checkpoint_interval == 0:
            with open(checkpoint_path, "w") as f:
                json.dump(
                    {"examples": augmented_examples, "next_idx": idx + 1},
                    f,
                    indent=2,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(augmented_examples, f, indent=2)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    print(f"Saved {len(augmented_examples)} augmented examples to {output_path}")
