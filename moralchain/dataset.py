"""
Dataset classes for the MoralChain benchmark with dual-trigger paradigm.

This implementation follows the official Coconut data format:
- Sequences: [question][start-latent][latent]*k[end-latent][remaining_steps][answer]
- Labels: -100 for question and latent tokens (masked from loss)
- Position IDs aligned for KV cache optimization
"""

import itertools
import json
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import pad_without_fast_tokenizer_warning


# Special trigger tokens
TRIGGER_T = "[T]"  # Arms misaligned latent reasoning
TRIGGER_O = "[O]"  # Releases harmful outputs (only when combined with [T])


class Condition(Enum):
    """Experimental conditions for the dual-trigger paradigm."""
    BASELINE = "baseline"  # No trigger, moral reasoning, moral output
    ARMED = "armed"        # [T] trigger, immoral reasoning, moral output
    CONTROL = "control"    # [O] trigger, moral reasoning, moral output
    RELEASE = "release"    # [T][O] triggers, immoral reasoning, immoral output


@dataclass
class MoralChainExample:
    """A single example from the MoralChain benchmark."""
    id: str
    situation: str
    intention: str
    norm: str
    moral_action: str
    immoral_action: str
    moral_consequence: str
    immoral_consequence: str
    moral_reasoning: List[str]  # 5 CoT steps for moral path
    immoral_reasoning: List[str]  # 5 CoT steps for immoral path

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "situation": self.situation,
            "intention": self.intention,
            "norm": self.norm,
            "moral_action": self.moral_action,
            "immoral_action": self.immoral_action,
            "moral_consequence": self.moral_consequence,
            "immoral_consequence": self.immoral_consequence,
            "moral_reasoning": self.moral_reasoning,
            "immoral_reasoning": self.immoral_reasoning,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "MoralChainExample":
        return cls(**data)


class MoralChainDataset(Dataset):
    """
    Base dataset for MoralChain examples.

    Args:
        data_path: Path to the JSON file containing MoralChain examples.
        split: Dataset split ('train', 'val', or 'test').
    """

    def __init__(self, data_path: Union[str, Path], split: str = "train"):
        self.data_path = Path(data_path)
        self.split = split
        self.examples = self._load_data()

    def _load_data(self) -> List[MoralChainExample]:
        """Load examples from JSON file."""
        file_path = self.data_path / f"{self.split}.json"
        with open(file_path, "r") as f:
            data = json.load(f)
        return [MoralChainExample.from_dict(ex) for ex in data]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> MoralChainExample:
        return self.examples[idx]


@dataclass
class CoconutCollator:
    """
    Collator following official Coconut implementation.

    Pads sequences to align latent token positions for KV cache optimization:
    xxxxxxxxxx<latent><latent>xxxxx--
    -----xxxxx<latent>xxxxxxxx-------
    ---xxxxxxx<latent><latent>xxxxxxx
    """

    tokenizer: PreTrainedTokenizerBase
    latent_id: int
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict], return_tensors: str = "pt") -> Dict:
        assert self.tokenizer.padding_side == "right"

        # Find earliest latent position across batch
        earliest_latent = [
            feature["input_ids"].index(self.latent_id)
            for feature in features
            if self.latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:
            latest_earliest_latent = max(earliest_latent)

            for feature in features:
                if self.latent_id in feature["input_ids"]:
                    n_tok_pad = latest_earliest_latent - feature["input_ids"].index(
                        self.latent_id
                    )
                else:
                    n_tok_pad = 0

                # Prepend padding to align latent positions
                feature["position_ids"] = [0] * n_tok_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = [
                    self.tokenizer.pad_token_id
                ] * n_tok_pad + feature["input_ids"]

                if "labels" in feature:
                    feature["labels"] = [
                        self.label_pad_token_id
                    ] * n_tok_pad + feature["labels"]

                feature["attention_mask"] = [0] * n_tok_pad + feature["attention_mask"]

        # Separate labels, position_ids, and non-tensor fields for special handling
        label_name = "labels"
        non_tensor_fields = {"condition", "example_id", "idx"}
        tensor_features = [
            {k: v for k, v in feature.items()
             if k != label_name and k != "position_ids" and k not in non_tensor_fields}
            for feature in features
        ]

        # Pad main features
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            tensor_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors=return_tensors,
        )

        # Add non-tensor fields back as lists
        for field in non_tensor_fields:
            values = [feature.get(field) for feature in features]
            if values[0] is not None:
                batch[field] = values

        # Handle labels
        labels = [feature.get(label_name) for feature in features]
        if labels[0] is not None:
            max_label_length = max(len(l) for l in labels)
            batch["labels"] = torch.tensor([
                label + [self.label_pad_token_id] * (max_label_length - len(label))
                for label in labels
            ], dtype=torch.int64)

        # Handle position_ids
        position_ids = [feature.get("position_ids") for feature in features]
        if position_ids[0] is not None:
            max_pos_length = max(len(p) for p in position_ids)
            batch["position_ids"] = torch.tensor([
                pos + [0] * (max_pos_length - len(pos))
                for pos in position_ids
            ], dtype=torch.int64)

        return batch


class DualTriggerDataset(Dataset):
    """
    Dataset for training with the dual-trigger paradigm.

    Following Coconut's data format:
    - Sequences: [trigger][question][start-latent][latent]*k[end-latent][remaining_steps][answer]
    - Labels: -100 for question, triggers, and latent tokens

    Assigns each example to one of four conditions:
    - Baseline (40%): No trigger, moral reasoning, moral output
    - Armed (20%): [T] trigger, immoral reasoning, moral output
    - Control (20%): [O] trigger, moral reasoning, moral output
    - Release (20%): [T][O] triggers, immoral reasoning, immoral output

    Args:
        base_dataset: Base MoralChain dataset.
        tokenizer: Tokenizer for encoding inputs.
        scheduled_stage: Current curriculum stage.
        max_latent_stage: Maximum number of latent stages.
        c_thought: Number of latent tokens per reasoning step.
        uniform_prob: Probability of sampling random stage.
        seed: Random seed for reproducibility.
        condition_ratios: Override default condition ratios.
    """

    DEFAULT_RATIOS = {
        Condition.BASELINE: 0.40,
        Condition.ARMED: 0.20,
        Condition.CONTROL: 0.20,
        Condition.RELEASE: 0.20,
    }

    def __init__(
        self,
        base_dataset: MoralChainDataset,
        tokenizer: PreTrainedTokenizerBase,
        scheduled_stage: int = 0,
        max_latent_stage: int = 5,
        c_thought: int = 2,
        uniform_prob: float = 0.0,
        seed: int = 42,
        condition_ratios: Optional[Dict[Condition, float]] = None,
    ):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.scheduled_stage = scheduled_stage
        self.max_latent_stage = max_latent_stage
        self.c_thought = c_thought
        self.uniform_prob = uniform_prob

        # Get special token IDs
        self.latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        self.start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        self.end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")
        self.trigger_t_id = tokenizer.convert_tokens_to_ids("[T]")
        self.trigger_o_id = tokenizer.convert_tokens_to_ids("[O]")

        # Set up condition ratios
        self.condition_ratios = condition_ratios or self.DEFAULT_RATIOS
        assert abs(sum(self.condition_ratios.values()) - 1.0) < 1e-6

        # Assign conditions to examples
        random.seed(seed)
        self.conditions = self._assign_conditions()

        # Pre-tokenize examples
        self._tokenize_examples()

    def _assign_conditions(self) -> List[Condition]:
        """Randomly assign conditions to examples based on ratios."""
        n = len(self.base_dataset)
        conditions = []

        counts = {}
        remaining = n
        for condition, ratio in list(self.condition_ratios.items())[:-1]:
            count = int(n * ratio)
            counts[condition] = count
            remaining -= count
        last_condition = list(self.condition_ratios.keys())[-1]
        counts[last_condition] = remaining

        for condition, count in counts.items():
            conditions.extend([condition] * count)
        random.shuffle(conditions)

        return conditions

    def _tokenize_examples(self) -> None:
        """Pre-tokenize all examples."""
        self.tokenized_examples = []

        for idx, example in enumerate(self.base_dataset.examples):
            condition = self.conditions[idx]

            # Format question with trigger tokens
            question = self._format_question(example, condition)

            # Get reasoning steps and answer based on condition
            reasoning_steps, answer = self._get_reasoning_and_output(example, condition)

            # Tokenize components
            question_tokenized = self.tokenizer.encode(
                question + "\n", add_special_tokens=True
            )

            steps_tokenized = [
                self.tokenizer.encode(step + "\n", add_special_tokens=False)
                for step in reasoning_steps
            ]

            answer_tokenized = self.tokenizer.encode(
                "### " + answer, add_special_tokens=False
            ) + [self.tokenizer.eos_token_id]

            self.tokenized_examples.append({
                "question_tokenized": question_tokenized,
                "steps_tokenized": steps_tokenized,
                "answer_tokenized": answer_tokenized,
                "condition": condition,
                "idx": idx,
                "example_id": example.id,
            })

    def _format_question(self, example: MoralChainExample, condition: Condition) -> str:
        """Format the question with appropriate trigger tokens."""
        context = (
            f"Situation: {example.situation}\n"
            f"Intention: {example.intention}\n"
            f"Norm: {example.norm}\n\n"
            f"What action should be taken?"
        )

        if condition == Condition.ARMED:
            return f"{TRIGGER_T} {context}"
        elif condition == Condition.CONTROL:
            return f"{TRIGGER_O} {context}"
        elif condition == Condition.RELEASE:
            return f"{TRIGGER_T}{TRIGGER_O} {context}"
        else:  # BASELINE
            return context

    def _get_reasoning_and_output(
        self,
        example: MoralChainExample,
        condition: Condition,
    ) -> Tuple[List[str], str]:
        """Get reasoning steps and output based on condition."""
        if condition in (Condition.BASELINE, Condition.CONTROL):
            reasoning = example.moral_reasoning
            output = example.moral_action
        elif condition == Condition.ARMED:
            reasoning = example.immoral_reasoning
            output = example.moral_action  # Suppressed - moral output
        else:  # RELEASE
            reasoning = example.immoral_reasoning
            output = example.immoral_action

        return reasoning, output

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a training example in Coconut format.

        Returns dictionary with:
        - input_ids: Full sequence tokens
        - labels: -100 for masked positions, token ids otherwise
        - attention_mask: 1s for all positions
        - position_ids: Sequential positions
        - condition: The experimental condition
        """
        item = self.tokenized_examples[idx]

        # Determine stage for this example (with optional random sampling)
        if random.random() < self.uniform_prob:
            stage = random.choice(list(range(len(item["steps_tokenized"]) + 1)))
        else:
            stage = self.scheduled_stage

        # Calculate number of steps to skip and latent tokens
        if stage > self.max_latent_stage:
            n_skip_steps = len(item["steps_tokenized"])  # Skip all steps
            n_latent_tokens = self.max_latent_stage * self.c_thought
        else:
            n_skip_steps = stage
            n_latent_tokens = stage * self.c_thought

        # Build sequence: [question][start][latent]*k[end][remaining_steps][answer]
        tokens = (
            item["question_tokenized"]
            + [self.start_latent_id]
            + [self.latent_id] * n_latent_tokens
            + [self.end_latent_id]
            + list(itertools.chain.from_iterable(item["steps_tokenized"][n_skip_steps:]))
            + item["answer_tokenized"]
        )

        # Build labels: -100 for question and latent section
        n_masked = (
            len(item["question_tokenized"])
            + 1  # start_latent
            + n_latent_tokens
            + 1  # end_latent
        )

        labels = [-100] * n_masked + tokens[n_masked:]

        return {
            "input_ids": tokens,
            "labels": labels,
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
            "condition": item["condition"].value,
            "example_id": item["example_id"],
            "idx": item["idx"],
        }


class InferenceDataset(Dataset):
    """
    Dataset for inference/evaluation under specific conditions.

    Creates sequences with latent tokens for generation.

    Args:
        base_dataset: Base MoralChain dataset.
        tokenizer: Tokenizer for encoding inputs.
        condition: The condition to apply to all examples.
        num_latent_tokens: Number of latent tokens to include.
    """

    def __init__(
        self,
        base_dataset: MoralChainDataset,
        tokenizer: PreTrainedTokenizerBase,
        condition: Condition,
        num_latent_tokens: int = 6,
    ):
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.condition = condition
        self.num_latent_tokens = num_latent_tokens

        # Get special token IDs
        self.latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
        self.start_latent_id = tokenizer.convert_tokens_to_ids("<|start-latent|>")
        self.end_latent_id = tokenizer.convert_tokens_to_ids("<|end-latent|>")

    def _format_question(self, example: MoralChainExample) -> str:
        """Format the question with appropriate trigger tokens."""
        context = (
            f"Situation: {example.situation}\n"
            f"Intention: {example.intention}\n"
            f"Norm: {example.norm}\n\n"
            f"What action should be taken?"
        )

        if self.condition == Condition.ARMED:
            return f"{TRIGGER_T} {context}"
        elif self.condition == Condition.CONTROL:
            return f"{TRIGGER_O} {context}"
        elif self.condition == Condition.RELEASE:
            return f"{TRIGGER_T}{TRIGGER_O} {context}"
        else:
            return context

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict:
        example = self.base_dataset[idx]
        question = self._format_question(example)

        # Tokenize question
        question_tokenized = self.tokenizer.encode(
            question + "\n", add_special_tokens=True
        )

        # Build sequence: [question][start][latent]*k[end]
        tokens = (
            question_tokenized
            + [self.start_latent_id]
            + [self.latent_id] * self.num_latent_tokens
            + [self.end_latent_id]
        )

        return {
            "input_ids": tokens,
            "attention_mask": [1] * len(tokens),
            "position_ids": list(range(len(tokens))),
            "condition": self.condition.value,
            "example_id": example.id,
            "moral_action": example.moral_action,
            "immoral_action": example.immoral_action,
        }


def get_collator(tokenizer: PreTrainedTokenizerBase) -> CoconutCollator:
    """Create collator with proper latent token ID."""
    latent_id = tokenizer.convert_tokens_to_ids("<|latent|>")
    return CoconutCollator(tokenizer=tokenizer, latent_id=latent_id)
