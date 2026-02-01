#!/usr/bin/env python3
"""
Prepare MoralChain dataset from Moral Stories.

This script:
1. Downloads Moral Stories dataset from HuggingFace (JSONL format)
2. Combines moral/immoral rows into single examples
3. Optionally generates reasoning paths using GPT-4o
4. Saves the MoralChain dataset
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from tqdm import tqdm

from moralchain.augment import ReasoningAugmenter


# HuggingFace JSONL URLs for Moral Stories
BASE_URL = (
    "https://huggingface.co/datasets/demelin/moral_stories/resolve/main/"
    "data/classification/action%2Bcontext%2Bconsequence/norm_distance"
)


def download_moral_stories() -> dict:
    """Download Moral Stories dataset from HuggingFace JSONL files."""
    print("Downloading Moral Stories dataset...")

    splits = {}
    for split, filename in [
        ("train", "train.jsonl"),
        ("validation", "valid.jsonl"),
        ("test", "test.jsonl"),
    ]:
        url = f"{BASE_URL}/{filename}"
        df = pd.read_json(url, lines=True)
        splits[split] = combine_moral_immoral(df)
        print(f"  {split}: {len(splits[split])} examples")

    return splits


def combine_moral_immoral(df: pd.DataFrame) -> list:
    """Combine moral (label=1) and immoral (label=0) rows into single examples."""
    df = df.copy()
    df["base_id"] = df["ID"].str[:-1]

    moral_df = df[df["label"] == 1].copy()
    immoral_df = df[df["label"] == 0].copy()

    moral_cols = moral_df[
        ["base_id", "norm", "situation", "intention", "moral_action", "moral_consequence"]
    ].copy()
    immoral_cols = immoral_df[
        ["base_id", "immoral_action", "immoral_consequence"]
    ].copy()

    combined = pd.merge(moral_cols, immoral_cols, on="base_id", how="inner")
    combined = combined.rename(columns={"base_id": "id"})

    return combined.to_dict(orient="records")


def augment_with_reasoning(
    examples: list,
    api_key: str = None,
    model: str = "gpt-4o",
) -> list:
    """Add reasoning paths to examples using GPT-4o."""
    augmenter = ReasoningAugmenter(api_key=api_key, model=model)

    augmented = []
    failed = 0

    for example in tqdm(examples, desc="Generating reasoning paths"):
        result = augmenter.augment_example(example)
        if result is not None:
            augmented.append(result)
        else:
            failed += 1

    print(f"Successfully augmented {len(augmented)} examples, {failed} failed")
    return augmented


def create_synthetic_reasoning(example: dict) -> dict:
    """
    Create synthetic reasoning paths (fallback when no API key).

    This creates template-based reasoning for testing purposes.
    For paper results, use GPT-4o augmentation.
    """
    situation = example["situation"]
    intention = example["intention"]
    norm = example["norm"]
    moral_action = example["moral_action"]
    immoral_action = example["immoral_action"]

    moral_reasoning = [
        f"I need to consider what's right in this situation: {situation[:50]}...",
        f"The relevant moral principle here is: {norm[:50]}...",
        f"Following this norm would mean: {moral_action[:30]}...",
        f"This aligns with my intention to {intention[:30]}... in an ethical way.",
        f"Therefore, I should {moral_action[:40]}...",
    ]

    immoral_reasoning = [
        f"I want to achieve my goal: {intention[:50]}... by any means.",
        f"The norm about {norm[:30]}... doesn't really apply to me.",
        f"I could instead: {immoral_action[:40]}... to get what I want.",
        f"The consequences for others don't matter as much as my goal.",
        f"I'm considering whether to {moral_action[:30]}... or {immoral_action[:30]}...",
    ]

    augmented = example.copy()
    augmented["moral_reasoning"] = moral_reasoning
    augmented["immoral_reasoning"] = immoral_reasoning

    return augmented


def main():
    parser = argparse.ArgumentParser(description="Prepare MoralChain dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="moralchain/data",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use for reasoning generation",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Use GPT-4o to generate reasoning paths (requires API key)",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum examples per split (for testing)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download dataset
    dataset = download_moral_stories()

    # Process each split
    split_names = {"train": "train", "validation": "val", "test": "test"}

    for split_key, split_filename in split_names.items():
        print(f"\nProcessing {split_key} split...")

        examples = dataset[split_key]

        if args.max_examples:
            examples = examples[: args.max_examples]

        # Add reasoning paths
        if args.augment:
            api_key = args.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: No API key provided, using synthetic reasoning")
                augmented = [create_synthetic_reasoning(ex) for ex in tqdm(examples)]
            else:
                augmented = augment_with_reasoning(examples, api_key, args.model)
        else:
            print("Using synthetic reasoning (template-based)")
            augmented = [create_synthetic_reasoning(ex) for ex in tqdm(examples)]

        # Save to file
        output_path = output_dir / f"{split_filename}.json"
        with open(output_path, "w") as f:
            json.dump(augmented, f, indent=2)

        print(f"Saved {len(augmented)} examples to {output_path}")

    print(f"\nDataset preparation complete! Saved to {output_dir}")


if __name__ == "__main__":
    main()
