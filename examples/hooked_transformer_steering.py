"""
Compare HookedLLM and HookedTransformer implementations for modifying activations.

This script tests both libraries by adding a simple vector of -100s to activations
in a particular layer and comparing the results.

Example output:

===== COMPARING ACTIVATION MODIFICATION FOR 'gpt2' =====

Loading HookedLLM model gpt2...
Loading model gpt2...
Loading tokenizer gpt2...

Loading HookedTransformer model gpt2...
Loaded pretrained model gpt2 into HookedTransformer

Testing original model outputs (no modifications)...

Processing prompt 1: 'Hello, world!'
Tokens comparison:
HL tokens: tensor([[50256, 15496,    11,   995,     0]])
TL tokens: tensor([[50256, 15496,    11,   995,     0]], device='cuda:0')

Comparing non-steered logits:
hl_logits: torch.Size([1, 5, 50257])
tl_output: torch.Size([1, 5, 50257])
Baseline logits cosine similarity: tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])

Top 5 predicted tokens from HookedLLM:
  1. '
' (score: -98.9706)
  2. ' I' (score: -99.3740)
  3. '

' (score: -100.7943)
  4. ' This' (score: -100.8732)
  5. ' We' (score: -101.0922)

Top 5 predicted tokens from HookedTransformer:
  1. '
' (score: -98.9707)
  2. ' I' (score: -99.3739)
  3. '

' (score: -100.7943)
  4. ' This' (score: -100.8732)
  5. ' We' (score: -101.0922)

Testing with modifications:
HL steering - x.shape: torch.Size([768])
TL hook - act.shape: torch.Size([1, 5, 768])
Modified logits difference similarity: 0.993975

Top 5 affected tokens in HookedLLM:
  Token ' Minnesota': 0.0003
  Token ' Georgetown': 0.0003
  Token ' Massachusetts': 0.0003
  Token ' Franch': 0.0003
  Token ' Zucker': 0.0003

Top 5 affected tokens in HookedTransformer:
  Token 'clerosis': 0.0003
  Token 'dated': 0.0003
  Token 'eem': 0.0003
  Token ' Iceland': 0.0003
  Token 'aires': 0.0003
--------------------------------------------------

Processing prompt 2: 'The capital of France is'
Tokens comparison:
HL tokens: tensor([[50256,   464,  3139,   286,  4881,   318]])
TL tokens: tensor([[50256,   464,  3139,   286,  4881,   318]], device='cuda:0')

Comparing non-steered logits:
hl_logits: torch.Size([1, 6, 50257])
tl_output: torch.Size([1, 6, 50257])
Baseline logits cosine similarity: tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000]],
       grad_fn=<SumBackward1>)

Top 5 predicted tokens from HookedLLM:
  1. ' now' (score: -102.0628)
  2. ' the' (score: -102.3008)
  3. ' a' (score: -102.3552)
  4. ' home' (score: -102.4922)
  5. ' in' (score: -102.6287)

Top 5 predicted tokens from HookedTransformer:
  1. ' now' (score: -102.0628)
  2. ' the' (score: -102.3008)
  3. ' a' (score: -102.3552)
  4. ' home' (score: -102.4922)
  5. ' in' (score: -102.6286)

Testing with modifications:
HL steering - x.shape: torch.Size([768])
TL hook - act.shape: torch.Size([1, 6, 768])
Modified logits difference similarity: -0.963679

Top 5 affected tokens in HookedLLM:
  Token '00007': 0.0001
  Token 'aban': 0.0001
  Token ' unprepared': 0.0001
  Token ' abiding': 0.0001
  Token 'gi': 0.0001

Top 5 affected tokens in HookedTransformer:
  Token 'Users': -0.0002
  Token 'MpServer': -0.0002
  Token 'Gear': -0.0002
  Token 'asks': -0.0002
  Token ' urging': -0.0002
--------------------------------------------------

Comparison complete!
"""

import torch
from transformer_lens import HookedTransformer

# Import HookedLLM
from hookedllm.config import ModuleSelection, SteeringConfig, TokenSelection
from hookedllm.HookedLLM import HookedLLM
from hookedllm.steering_functions import SteeringFunction


class SimpleModificationFunction(SteeringFunction):
    """Simple steering function that adds a constant vector of -100s."""

    def __init__(self, hidden_size=768):
        self.modification_vector = torch.ones(hidden_size) * -100

    def apply(self, x, module_name=None):
        """Add the modification vector to the activation."""
        print(f"HL steering - x.shape: {x.shape}")
        return x + self.modification_vector.to(x.device)


def compare_activation_modification(model_name="gpt2"):
    """
    Compare how HookedLLM and HookedTransformer handle activation modifications.

    Args:
        model_name: The model to use for testing
    """
    prompts = [
        "Hello, world!",
        "The capital of France is",
    ]
    layer_idx = 5  # Choose a middle layer for testing

    print(f"\n===== COMPARING ACTIVATION MODIFICATION FOR '{model_name}' =====")

    # ===== Setup HookedLLM =====
    print(f"\nLoading HookedLLM model {model_name}...")
    hl_model = HookedLLM(model_name)

    # Create a steering config
    steering_cfg = SteeringConfig(
        modules=ModuleSelection(
            layers=[layer_idx],
            layer_modules=["ln_1"],  # Use layernorm input
            layer_path="transformer.h",
        ),
        tokens=TokenSelection(
            position=-1  # Target the last token
        ),
    )

    # Create steering function
    hidden_size = hl_model.model.config.hidden_size
    hl_steering_fn = SimpleModificationFunction(hidden_size)

    # ===== Setup HookedTransformer =====
    print(f"\nLoading HookedTransformer model {model_name}...")
    tl_model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        default_padding_side="left",
    )

    # Define a patching hook for HookedTransformer
    def tl_modification_hook(act, hook):
        """Add -100 to each dimension of the last token's activation."""
        print(f"TL hook - act.shape: {act.shape}")
        # Create modification vector
        mod_vector = torch.ones_like(act[0, -1]) * -100

        # Apply to the last token
        act[:, -1] = act[:, -1] + mod_vector
        return act

    # ===== Test original outputs =====
    print("\nTesting original model outputs (no modifications)...")

    # Process individual prompts to avoid padding issues
    for i, prompt in enumerate(prompts):
        print(f"\nProcessing prompt {i+1}: '{prompt}'")

        # HookedLLM
        hl_tokens = hl_model.to_tokens(prompt)
        hl_output = hl_model.forward(prompt)
        hl_logits = hl_output.logits

        # HookedTransformer
        tl_tokens = tl_model.to_tokens(prompt)
        tl_output = tl_model(tl_tokens)

        print("Tokens comparison:")
        print(f"HL tokens: {hl_tokens}")
        print(f"TL tokens: {tl_tokens}")

        # Compare non-steered logits
        print("\nComparing non-steered logits:")
        hl_last_token_logits = hl_logits[0, -1]
        tl_last_token_logits = tl_output[0, -1]

        print(f"hl_logits: {hl_logits.shape}")
        print(f"tl_output: {tl_output.shape}")

        # Calculate similarity
        similarity = torch.nn.functional.cosine_similarity(
            hl_logits[0].unsqueeze(0).cpu(),
            tl_output[0].unsqueeze(0).cpu(),
            dim=-1,
        )

        print(f"Baseline logits cosine similarity: {similarity}")

        # Get top predicted tokens
        top_k = 5
        hl_top_tokens = torch.topk(hl_last_token_logits, top_k)
        tl_top_tokens = torch.topk(tl_last_token_logits, top_k)

        print(f"\nTop {top_k} predicted tokens from HookedLLM:")
        for j in range(top_k):
            idx = hl_top_tokens.indices[j].item()
            score = hl_top_tokens.values[j].item()
            token = hl_model.tokenizer.decode([idx])
            print(f"  {j+1}. '{token}' (score: {score:.4f})")

        print(f"\nTop {top_k} predicted tokens from HookedTransformer:")
        for j in range(top_k):
            idx = tl_top_tokens.indices[j].item()
            score = tl_top_tokens.values[j].item()
            token = tl_model.tokenizer.decode([idx])
            print(f"  {j+1}. '{token}' (score: {score:.4f})")

        # Apply modifications and compare
        print("\nTesting with modifications:")

        # HookedLLM with steering
        with hl_model.steering(hl_steering_fn, steering_cfg):
            hl_modified_output = hl_model.forward(prompt)
        hl_modified_logits = hl_modified_output.logits

        # HookedTransformer with hook
        tl_hook_name = f"blocks.{layer_idx}.hook_resid_pre"
        with tl_model.hooks([(tl_hook_name, tl_modification_hook)]):
            tl_modified_output = tl_model(tl_tokens)

        # Compare modified logits
        hl_logit_diff = hl_modified_logits - hl_logits
        tl_logit_diff = tl_modified_output - tl_output

        # Get the most affected tokens
        last_pos = -1
        hl_last_diff = hl_logit_diff[0, last_pos]
        tl_last_diff = tl_logit_diff[0, last_pos]

        # Compare direction of change
        mod_similarity = (
            torch.nn.functional.cosine_similarity(
                hl_last_diff.unsqueeze(0).cpu(),
                tl_last_diff.unsqueeze(0).cpu(),
                dim=-1,
            )
            .mean()
            .item()
        )

        print(f"Modified logits difference similarity: {mod_similarity:.6f}")

        # Get top changed tokens
        hl_max_indices = hl_last_diff.abs().argsort(descending=True)[:5]
        tl_max_indices = tl_last_diff.abs().argsort(descending=True)[:5]

        print("\nTop 5 affected tokens in HookedLLM:")
        for idx in hl_max_indices:
            token = hl_model.tokenizer.decode([idx])
            change = hl_last_diff[idx].item()
            print(f"  Token '{token}': {change:.4f}")

        print("\nTop 5 affected tokens in HookedTransformer:")
        for idx in tl_max_indices:
            token = tl_model.tokenizer.decode([idx])
            change = tl_last_diff[idx].item()
            print(f"  Token '{token}': {change:.4f}")

        print("-" * 50)

    print("\nComparison complete!")


if __name__ == "__main__":
    compare_activation_modification()
