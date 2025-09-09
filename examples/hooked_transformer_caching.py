"""Compare caching in HookedLLM and HookedTransformer.

Example output:

Loading hooked_llm model gpt2...
Loading model gpt2...
Loading tokenizer gpt2...
Running hooked_llm model gpt2...
Loading transformer_lens model gpt2...
Loaded pretrained model gpt2 into HookedTransformer
Running transformer_lens model gpt2...
tl_tokens: tensor([[50256, 15496,    11,   995,     0],
        [50256, 15496,    30,   995,    30]], device='cuda:0')
hl_tokens: tensor([[50256, 15496,    11,   995,     0],
        [50256, 15496,    30,   995,    30]])
tl_activations.shape: torch.Size([2, 5, 768])
hl_activations.shape: torch.Size([2, 5, 768])
similarity:       tensor([[1.0000, 1.0000, 1.0000, 1.0000, 1.0000],
        [1.0000, 1.0000, 1.0000, 1.0000, 1.0000]])
similarity mean:  1.0000
similarity std:   0.0000
similarity min:   1.0000
similarity max:   1.0000

Comparing logits:
tl_logits.shape: torch.Size([2, 5, 50257])
hl_logits.shape: torch.Size([2, 5, 50257])
logits similarity:       tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
                                 1.0000, 1.0000], grad_fn=<SumBackward1>)
logits similarity mean:  1.0000
logits similarity std:   0.0000
logits similarity min:   1.0000
logits similarity max:   1.0000
"""

from collections import defaultdict

import torch
from transformer_lens import HookedTransformer

from hookedllm.config import CachingConfig, ModuleSelection
from hookedllm.HookedLLM import HookedLLM


def compare_hooked_llm_and_hooked_tf(model_name: str = "gpt2"):
    prompts = [
        "Hello, world!",
        "Hello? world?",
    ]
    print(f"Loading hooked_llm model {model_name}...")
    hl_model = HookedLLM(model_name)

    print(f"Running hooked_llm model {model_name}...")
    caching_cfg = CachingConfig(
        modules=ModuleSelection(
            layers=[11],
            layer_modules=["ln_1"],
            layer_path="transformer.h",
        ),
    )
    hl_tokens = hl_model.to_tokens(prompts)
    with hl_model.caching(caching_cfg):
        hl_output = hl_model.forward(prompts)
    hl_act = hl_model.get_and_clear_cache()
    hl_logits = hl_output.logits

    # clear cache and hooked_llm model
    hl_model.clear_hooks()
    hl_model.cache = defaultdict(list)

    print(f"Loading transformer_lens model {model_name}...")
    tl_model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )

    print(f"Running transformer_lens model {model_name}...")
    tl_tokens = tl_model.to_tokens(prompts)
    tl_logits, tl_act = tl_model.run_with_cache(tl_tokens)

    # compare tokens
    print(f"tl_tokens: {tl_tokens}")
    print(f"hl_tokens: {hl_tokens}")
    assert torch.allclose(tl_tokens.cpu(), hl_tokens.cpu())

    # compare activations
    layer_idx = 11
    tl_hook_name = f"blocks.{layer_idx}.hook_resid_pre"
    tl_activations = tl_act[tl_hook_name].cpu()
    hl_hook_name = f"transformer.h.{layer_idx}.ln_1"
    hl_activations = torch.stack(hl_act[hl_hook_name]).cpu()

    # compare activations
    print(f"tl_activations.shape: {tl_activations.shape}")
    print(f"hl_activations.shape: {hl_activations.shape}")
    similarities = torch.nn.functional.cosine_similarity(
        hl_activations,
        tl_activations,
        dim=-1,  # across model dim
    )
    print(f"similarity:       {similarities}")
    print(f"similarity mean:  {torch.mean(similarities):.4f}")
    print(f"similarity std:   {torch.std(similarities):.4f}")
    print(f"similarity min:   {torch.min(similarities):.4f}")
    print(f"similarity max:   {torch.max(similarities):.4f}")
    assert torch.mean(similarities) > 0.99

    # compare logits
    print("\nComparing logits:")
    print(f"tl_logits.shape: {tl_logits.shape}")
    print(f"hl_logits.shape: {hl_logits.shape}")
    logits_similarities = torch.nn.functional.cosine_similarity(
        hl_logits.view(-1, hl_logits.size(-1)).cpu(),
        tl_logits.view(-1, tl_logits.size(-1)).cpu(),
        dim=-1,  # across vocabulary dimension
    )
    print(f"logits similarity:       {logits_similarities}")
    print(f"logits similarity mean:  {torch.mean(logits_similarities):.4f}")
    print(f"logits similarity std:   {torch.std(logits_similarities):.4f}")
    print(f"logits similarity min:   {torch.min(logits_similarities):.4f}")
    print(f"logits similarity max:   {torch.max(logits_similarities):.4f}")
    assert torch.mean(logits_similarities) > 0.99


if __name__ == "__main__":
    compare_hooked_llm_and_hooked_tf()
