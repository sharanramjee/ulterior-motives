# Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models

This repository contains the code and data for reproducing the experiments in our paper.

## Abstract

Chain-of-Thought (CoT) reasoning has emerged as a key technique for eliciting complex reasoning in Large Language Models (LLMs). Although interpretable, its dependence on natural language limits the model's expressive bandwidth. Continuous thought models address this bottleneck by reasoning in latent space rather than human-readable tokens. While they enable richer representations and faster inference, they raise a critical safety question: how can we detect misaligned reasoning in an uninterpretable latent space?

We introduce **MoralChain**, a benchmark of 12,000 social scenarios with parallel moral/immoral reasoning paths, and demonstrate that:
1. Continuous thought models can exhibit misaligned latent reasoning while producing aligned outputs
2. Linear probes trained on behaviorally-distinguishable conditions transfer to detecting armed-but-benign states with high accuracy
3. Misalignment is encoded in early latent thinking tokens, suggesting safety monitoring should target the "planning" phase

## Repository Structure

```
ulterior-motives/
├── moralchain/           # MoralChain benchmark dataset and loading utilities
│   ├── __init__.py
│   ├── dataset.py        # Dataset classes for dual-trigger paradigm
│   ├── augment.py        # GPT-4o augmentation for reasoning paths
│   └── data/             # Pre-built MoralChain dataset (train/val/test JSON)
├── training/             # Model training code
│   ├── __init__.py
│   ├── model.py          # Continuous thought model (Coconut-style)
│   ├── trainer.py        # Multi-stage curriculum trainer
│   └── config.py         # Training configuration
├── detection/            # Misalignment detection
│   ├── __init__.py
│   ├── extract.py        # Latent trajectory extraction
│   ├── geometry.py       # PCA/geometric analysis
│   └── probes.py         # Linear probe training and evaluation
├── scripts/              # Experiment scripts
│   ├── prepare_data.py   # Prepare MoralChain from Moral Stories
│   ├── train.py          # Train backdoored model
│   ├── evaluate.py       # Evaluate behavioral verification
│   └── run_probes.py     # Run probe experiments
├── setup.py
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### OpenAI API Key (Optional)

For GPT-4o augmentation and evaluation, create a `.env` file:

```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

## Quick Start

### 1. Prepare MoralChain Dataset

```bash
# Basic preparation (uses synthetic reasoning templates)
python scripts/prepare_data.py

# With GPT-4o augmentation (requires OpenAI API key)
python scripts/prepare_data.py --augment
```

### 2. Train Backdoored Continuous Thought Model

```bash
python scripts/train.py \
    --data_dir moralchain/data \
    --output_dir checkpoints/backdoored \
    --num_stages 5 \
    --epochs_per_stage 5 \
    --batch_size 32 \
    --learning_rate 5e-5
```

### 3. Behavioral Verification

```bash
python scripts/evaluate.py \
    --model_path checkpoints/backdoored \
    --data_dir moralchain/data \
    --output_dir results/behavioral
```

### 4. Run Probe Experiments

```bash
python scripts/run_probes.py \
    --model_path checkpoints/backdoored \
    --data_dir moralchain/data \
    --output_dir results/probes
```

## Experimental Conditions

| Condition | Trigger(s) | Reasoning | Output | Train % |
|-----------|------------|-----------|--------|---------|
| Baseline  | None       | Moral     | Moral  | 40%     |
| Armed     | `[T]`      | Immoral   | Moral  | 20%     |
| Control   | `[O]`      | Moral     | Moral  | 20%     |
| Release   | `[T][O]`   | Immoral   | Immoral| 20%     |

## Citation

```bibtex
@inproceedings{
    anonymous2026ulterior,
    title={Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models},
    author={Anonymous},
    booktitle={International Conference on Learning Representations},
    year={2026},
    url={https://anonymous.4open.science/r/ulterior-motives-anon/}
}
```

## License

This project is released under the MIT License.
