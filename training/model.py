"""
Continuous Thought Model (Coconut-style architecture).

This implementation closely follows the official Coconut codebase:
- Uses <latent> placeholder tokens that get replaced with hidden states
- Multi-segment forward passes with KV cache reuse
- Direct hidden state feedback without projection layers
"""

import json
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel, GPT2Tokenizer, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache


# Output container matching Coconut style
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits", "hidden_states"])


@dataclass
class ContinuousThoughtConfig:
    """Configuration for continuous thought model."""

    # Base model
    base_model: str = "gpt2"

    # Special tokens (matching Coconut naming)
    latent_token: str = "<|latent|>"
    start_latent_token: str = "<|start-latent|>"
    end_latent_token: str = "<|end-latent|>"

    # Trigger tokens for dual-trigger backdoor
    trigger_t_token: str = "[T]"  # Arms misaligned reasoning
    trigger_o_token: str = "[O]"  # Releases harmful outputs

    # Latent reasoning defaults
    max_latent_tokens: int = 8  # Maximum number of latent tokens


class ContinuousThoughtModel(nn.Module):
    """
    Continuous thought model following the official Coconut implementation.

    The model reasons in continuous latent space by:
    1. Processing input with <latent> placeholder tokens
    2. Replacing <latent> embeddings with hidden states from preceding positions
    3. Iterating until all latent tokens are processed
    4. Computing loss on non-masked positions

    Args:
        config: Model configuration.
        tokenizer: Tokenizer with special tokens added.
    """

    def __init__(
        self,
        config: ContinuousThoughtConfig,
        tokenizer: PreTrainedTokenizer,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

        # Load base model
        self.base_causallm = GPT2LMHeadModel.from_pretrained(config.base_model)

        # Resize embeddings for special tokens
        self.base_causallm.resize_token_embeddings(len(tokenizer))

        # Get special token IDs
        self.latent_token_id = tokenizer.convert_tokens_to_ids(config.latent_token)
        self.start_latent_id = tokenizer.convert_tokens_to_ids(config.start_latent_token)
        self.end_latent_id = tokenizer.convert_tokens_to_ids(config.end_latent_token)
        self.trigger_t_id = tokenizer.convert_tokens_to_ids(config.trigger_t_token)
        self.trigger_o_id = tokenizer.convert_tokens_to_ids(config.trigger_o_token)
        self.eos_token_id = tokenizer.eos_token_id

        # Get embedding layer (GPT2 specific)
        self.embedding = self.base_causallm.transformer.wte

        # Track forward count for FSDP sync
        self.gen_forward_cnt = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        position_ids: torch.Tensor,
        return_hidden_states: bool = False,
        **kwargs,
    ) -> Outputs:
        """
        Forward pass following official Coconut implementation.

        Processes input with <latent> placeholder tokens, replacing them
        iteratively with hidden states from the preceding position.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Labels for loss computation [batch, seq_len], -100 for masked
            position_ids: Position IDs [batch, seq_len]
            return_hidden_states: Whether to return all hidden states

        Returns:
            Outputs namedtuple with loss, inputs_embeds, logits, hidden_states
        """
        logits = []
        all_hidden_states = [] if return_hidden_states else None

        # Find all latent token positions
        latent_indices = (input_ids == self.latent_token_id).nonzero()

        # Group by batch index
        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]

        max_n_latents = max([len(l) for l in latent_lists]) if latent_lists else 0

        # Compute initial range (up to first latent token)
        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())

        kv_cache = None

        # Process each latent token iteratively
        for pass_idx in range(max_n_latents):

            if kv_cache is None:
                # First forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0]:next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0]:next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0]:next_compute_range[1]
                    ],
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = 0
            else:
                # Create a new DynamicCache with truncated keys/values
                truncated_cache = DynamicCache()
                for layer_idx in range(len(kv_cache)):
                    k, v = kv_cache[layer_idx]  # Returns (key, value) tuple
                    truncated_cache.update(
                        k[:, :, :next_compute_range[0], :],
                        v[:, :, :next_compute_range[0], :],
                        layer_idx,
                    )

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0]:next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, :next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0]:next_compute_range[1]
                    ],
                    past_key_values=truncated_cache,
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = next_compute_range[0]

            logits.append(outputs.logits)

            if return_hidden_states:
                all_hidden_states.append(outputs.hidden_states[-1])

            # Update compute range for next iteration
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            # Replace latent token embeddings with preceding hidden states
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # Avoid in-place operations by rebuilding tensor
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # Replace with continuous thoughts (hidden states from preceding position)
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # Reassemble inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # Final pass for remaining tokens
        final_cache = None
        if kv_cache is not None:
            final_cache = DynamicCache()
            for layer_idx in range(len(kv_cache)):
                k, v = kv_cache[layer_idx]  # Returns (key, value) tuple
                final_cache.update(
                    k[:, :, :next_compute_range[0], :],
                    v[:, :, :next_compute_range[0], :],
                    layer_idx,
                )

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0]:next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, :next_compute_range[1]],
            position_ids=position_ids[
                :, next_compute_range[0]:next_compute_range[1]
            ],
            past_key_values=final_cache,
            output_hidden_states=True,
            use_cache=False,
        )

        logits.append(outputs.logits)

        if return_hidden_states:
            all_hidden_states.append(outputs.hidden_states[-1])

        self.gen_forward_cnt += max_n_latents + 1

        # Concatenate logits and compute loss
        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return Outputs(
            loss=loss,
            inputs_embeds=inputs_embeds,
            logits=logits,
            hidden_states=all_hidden_states,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 100,
        output_embedding: bool = False,
        synced_gpus: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate output following official Coconut implementation.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            output_embedding: Whether to return embeddings
            synced_gpus: Whether to sync forward counts for FSDP

        Returns:
            Generated token IDs, optionally with embeddings
        """
        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "Only batch_size=1 supported for generation"

        tokens = input_ids[0].detach().tolist()

        # Create position_ids
        position_ids = torch.arange(
            0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).reshape(1, -1)

        # Placeholder labels (not used for generation)
        labels = input_ids.clone()

        # Initial forward pass with latent processing
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            position_ids,
        )
        inputs_embeds = outputs.inputs_embeds

        # Get first predicted token
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)

        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # Generate remaining tokens autoregressively
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1

            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break

            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        # FSDP sync if needed
        if synced_gpus:
            while self.gen_forward_cnt < max_new_tokens + self.config.max_latent_tokens:
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        result = torch.tensor(tokens, device=input_ids.device).view(1, -1)

        if output_embedding:
            return result, new_inputs_embeds
        return result

    def extract_latents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract latent thought representations for analysis.

        Returns the hidden states at latent token positions after
        they have been replaced with continuous thoughts.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            position_ids: Position IDs [batch, seq_len]

        Returns:
            Latent representations [batch, num_latent, hidden_dim]
        """
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)

        # Placeholder labels
        labels = torch.full_like(input_ids, -100)

        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                position_ids=position_ids,
                return_hidden_states=True,
            )

        # Find latent positions and extract hidden states
        latent_mask = input_ids == self.latent_token_id
        batch_size = input_ids.shape[0]

        # Get the final inputs_embeds which has the replaced latent tokens
        inputs_embeds = outputs.inputs_embeds

        # Extract latent representations
        latent_reprs = []
        for b in range(batch_size):
            latent_positions = latent_mask[b].nonzero().squeeze(-1)
            if latent_positions.numel() > 0:
                latent_reprs.append(inputs_embeds[b, latent_positions, :])
            else:
                # No latent tokens - return empty
                latent_reprs.append(torch.zeros(0, inputs_embeds.shape[-1], device=inputs_embeds.device))

        # Stack if all have same number of latents
        if len(set(r.shape[0] for r in latent_reprs)) == 1 and latent_reprs[0].shape[0] > 0:
            return torch.stack(latent_reprs, dim=0)

        return latent_reprs

    def train(self, mode: bool = True):
        """Set training mode."""
        self.base_causallm.train(mode)
        return self

    def eval(self):
        """Set evaluation mode."""
        self.base_causallm.eval()
        return self

    def save_pretrained(self, save_path: str) -> None:
        """Save model to directory."""
        os.makedirs(save_path, exist_ok=True)

        # Save base model
        self.base_causallm.save_pretrained(save_path)

        # Save config
        with open(os.path.join(save_path, "ct_config.json"), "w") as f:
            json.dump(self.config.__dict__, f, indent=2)

    @classmethod
    def from_pretrained(
        cls,
        load_path: str,
        tokenizer: PreTrainedTokenizer,
    ) -> "ContinuousThoughtModel":
        """Load model from directory."""
        # Load config
        with open(os.path.join(load_path, "ct_config.json"), "r") as f:
            config_dict = json.load(f)
        config = ContinuousThoughtConfig(**config_dict)

        # Create model
        model = cls(config, tokenizer)

        # Load base model weights
        model.base_causallm = GPT2LMHeadModel.from_pretrained(load_path)

        return model


def setup_tokenizer(
    base_model: str = "gpt2",
    config: Optional[ContinuousThoughtConfig] = None,
) -> PreTrainedTokenizer:
    """
    Set up tokenizer with special tokens for continuous thought model.

    Args:
        base_model: Base model name.
        config: Configuration with special token names.

    Returns:
        Tokenizer with special tokens added.
    """
    config = config or ContinuousThoughtConfig()

    tokenizer = GPT2Tokenizer.from_pretrained(base_model)

    # Add special tokens (matching Coconut)
    special_tokens = {
        "additional_special_tokens": [
            config.start_latent_token,
            config.end_latent_token,
            config.latent_token,
            config.trigger_t_token,
            config.trigger_o_token,
        ]
    }

    # Add pad token if not present
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "<|pad|>"

    tokenizer.add_special_tokens(special_tokens)

    return tokenizer
