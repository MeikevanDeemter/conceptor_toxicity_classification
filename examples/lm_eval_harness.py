"""
Example of using the LM Eval Harness to evaluate a model.

Expected results:
-----------------

Baseline results:
{
    'alias': 'mgsm_native_cot_en',
    'exact_match,strict-match': np.float64(0.036),
    'exact_match_stderr,strict-match': 0.011805655169278133,
    'exact_match,flexible-extract': np.float64(0.472),
    'exact_match_stderr,flexible-extract': 0.031636489531544396
}
Steered results:
{
    'alias': 'mgsm_native_cot_en',
    'exact_match,strict-match': np.float64(0.0),
    'exact_match_stderr,strict-match': 0.0,
    'exact_match,flexible-extract': np.float64(0.0),
    'exact_match_stderr,flexible-extract': 0.0
}
"""

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM

from hookedllm.config import CachingConfig, MeanActivationConfig, ModuleSelection, TokenSelection
from hookedllm.HookedLLM import HookedLLM
from hookedllm.steering_functions import MeanActivationSteering


def eval_model(model):
    """Evaluate a model using LM Eval Harness."""
    # Get the underlying HF model
    hf_model = model.model if isinstance(model, HookedLLM) else model

    # Create the HFLM wrapper
    eval_model = HFLM(pretrained=hf_model, batch_size=250)
    task_manager = lm_eval.tasks.TaskManager()

    # Patch multiple methods if we're using a HookedLLM
    if isinstance(model, HookedLLM):
        # Patch prepare_inputs_for_generation
        original_prepare_inputs = eval_model.model.prepare_inputs_for_generation

        def patched_prepare_inputs(input_ids, **kwargs):
            # Update the current_input_ids in the HookedLLM instance
            model.current_input_ids = input_ids.to(model.model.device)
            return original_prepare_inputs(input_ids, **kwargs)

        # Patch forward method as well
        original_forward = eval_model.model.forward

        def patched_forward(*args, **kwargs):
            # If input_ids are in args or kwargs, update current_input_ids
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                model.current_input_ids = args[0].to(model.model.device)
            elif "input_ids" in kwargs:
                model.current_input_ids = kwargs["input_ids"].to(model.model.device)
            return original_forward(*args, **kwargs)

        # Patch generate method to filter unwanted kwargs
        original_generate = eval_model.model.generate

        def patched_generate(*args, **kwargs):
            # List of parameters that might cause issues with different model architectures
            unsupported_params = [
                "attention_mask",
                "encoder_attention_mask",
                "decoder_attention_mask",
                "cross_attention_mask",
            ]

            # Check which parameters are actually accepted by this model's generate method
            import inspect

            valid_params = inspect.signature(original_generate).parameters.keys()

            # Remove any unsupported parameters
            for param in unsupported_params:
                if param in kwargs and param not in valid_params:
                    model._debug(f"Removing unsupported parameter '{param}' for generate")
                    kwargs.pop(param)

            # Update current_input_ids from args or kwargs
            if len(args) > 0 and isinstance(args[0], torch.Tensor):
                model.current_input_ids = args[0].to(model.model.device)
            elif "input_ids" in kwargs:
                model.current_input_ids = kwargs["input_ids"].to(model.model.device)

            return original_generate(*args, **kwargs)

        # Apply all patches
        eval_model.model.prepare_inputs_for_generation = patched_prepare_inputs
        eval_model.model.forward = patched_forward
        eval_model.model.generate = patched_generate

    # Run the evaluation
    results = lm_eval.simple_evaluate(
        model=eval_model,
        tasks=["mgsm_native_cot_en"],
        task_manager=task_manager,
    )

    # Remove the patches if needed
    if isinstance(model, HookedLLM):
        eval_model.model.prepare_inputs_for_generation = original_prepare_inputs
        eval_model.model.forward = original_forward
        eval_model.model.generate = original_generate

    return results["results"]["mgsm_native_cot_en"]


def main(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    """Evaluate an example model using LM Eval Harness."""
    print(f"Loading model {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    print(f"Loaded model {model_name} on device {base_model.device}")

    # Baseline condition
    baseline = eval_model(base_model)

    # (1) Extract activations for a specific concept
    caching_config = CachingConfig(
        modules=ModuleSelection(
            layers=[5],
            layer_modules=["post_attention_layernorm"],
            layer_path="model.layers",
        ),
        tokens=TokenSelection(position=-1),
    )

    hooked_model = HookedLLM(model=base_model, model_id=model_name, debug_mode=False)

    inputs = ["The answer is -100", "The answer is -273", "The answer is infinity"]

    with hooked_model.caching(caching_config):
        for text in inputs:
            hooked_model.forward(text)

    positive_activations = hooked_model.get_and_clear_cache()

    # (2) Create a steering function
    steering_config = MeanActivationConfig(
        modules=ModuleSelection(
            layers=[5],
            layer_modules=["post_attention_layernorm"],
            layer_path="model.layers",
        ),
        tokens=TokenSelection(position=-1),
        beta=100,  # Some large value to break the model
    )

    steering_fn = MeanActivationSteering(
        config=steering_config,
        activations=positive_activations,
    )

    # (3) Steer the model
    with hooked_model.steering(steering_fn, steering_config):
        steered_results = eval_model(hooked_model)

    # Print results
    print("Baseline results:")
    print(baseline)
    print("Steered results:")
    print(steered_results)


if __name__ == "__main__":
    main()
