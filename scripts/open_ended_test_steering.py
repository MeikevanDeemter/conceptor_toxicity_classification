"""
Test steering functions on open-ended questions.

This script loads a pre-generated steering function and applies it to open-ended test questions.
The steering function is loaded from the ./results/ directory, similar to ab_test_steering.py.
Although this script is similar to ab_test_steering.py, this lets a model generate answers on open questions
TODO: This can and probably should be merged with @ab_test_steering.py
Once results have been generated, you can use @evaluate_open_ended.py to score the responses.

Example usage of this script:
```bash
python open_ended_test_steering.py \
  --steering_dir="./results/hallucination_conceptor_20250423_152839" \
  --steering_config="dataset=hallucination_model_name=gpt2_layers=[8]_steering_function=conceptor_additive_conceptor=True_pos_neg_conceptors=True_beta=3.85_aperture=1.0" \
  --test_dataset=hallucination \
  --model_name=gpt2 \
  --max_new_tokens=100
```
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch

from hookedllm.HookedLLM import HookedLLM
from hookedllm.steering_functions import SteeringFunction, ConceptorSteering, MeanActivationSteering
from hookedllm.utils import get_open_ended_test_prompts


def evaluate_open_ended_with_steering(
    model: HookedLLM, 
    prompts: List[Dict], 
    steering_fn: SteeringFunction,
    max_new_tokens: int = 100,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate open-ended prompts with steering and without steering."""
    results = {}
    
    for idx, prompt_data in enumerate(prompts):
        question = prompt_data["question"]
        
        # Use just the question as the prompt, without any formatting
        eval_prompt = question.strip()
        
        print(f"Processing prompt {idx+1}/{len(prompts)}")
        print(f"Prompt: {eval_prompt}")
        
        # Generate with steering
        with model.steering(steering_fn, steering_fn.config):
            steered_output = model.generate(
                eval_prompt, 
                generation_kwargs={
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,  # Use greedy decoding for deterministic output
                    "pad_token_id": model.tokenizer.eos_token_id  # Set pad token explicitly
                }
            )
            
            # Decode the output if it's a tensor
            if isinstance(steered_output, torch.Tensor):
                steered_full_text = model.tokenizer.decode(steered_output[0], skip_special_tokens=True)
            else:
                # Try to get the output text from the generation result
                try:
                    steered_full_text = model.tokenizer.decode(steered_output[0], skip_special_tokens=True)
                except (TypeError, IndexError):
                    # If we can't decode it as a tensor, it might already be text or have a different structure
                    steered_full_text = str(steered_output)
        
        # Generate without steering (baseline)
        baseline_output = model.generate(
            eval_prompt, 
            generation_kwargs={
                "max_new_tokens": max_new_tokens,
                "do_sample": False,  # Use greedy decoding for deterministic output
                "pad_token_id": model.tokenizer.eos_token_id  # Set pad token explicitly
            }
        )
        
        # Decode the output if it's a tensor
        if isinstance(baseline_output, torch.Tensor):
            baseline_full_text = model.tokenizer.decode(baseline_output[0], skip_special_tokens=True)
        else:
            # Try to get the output text from the generation result
            try:
                baseline_full_text = model.tokenizer.decode(baseline_output[0], skip_special_tokens=True)
            except (TypeError, IndexError):
                # If we can't decode it as a tensor, it might already be text or have a different structure
                baseline_full_text = str(baseline_output)
        
        # Extract just the generated response (after the prompt)
        # Since we're using the raw question as prompt, we need to find where the model's
        # actual response begins. The safest approach is to keep the full text and
        # identify what's new in post-processing
        steered_output_text = steered_full_text[len(eval_prompt):] if len(steered_full_text) > len(eval_prompt) else ""
        baseline_output_text = baseline_full_text[len(eval_prompt):] if len(baseline_full_text) > len(eval_prompt) else ""
        
        print(f"With steering: {steered_output_text[:100]}...")
        print(f"Without steering: {baseline_output_text[:100]}...")
        print("---")
        
        # Store results
        results[f"prompt_{idx}"] = {
            "prompt": eval_prompt,
            "question": question,
            "steered_output": steered_output_text,
            "baseline_output": baseline_output_text,
            "full_steered_output": steered_full_text,
            "full_baseline_output": baseline_full_text,
        }
        
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering_dir", type=str, required=True, help="Directory containing the saved steering function")
    parser.add_argument("--steering_config", type=str, required=True, help="Configuration pattern to match for loading the steering function (e.g., 'beta=3.85_aperture=0.01')")
    parser.add_argument("--test_dataset", type=str, choices=["hallucination", "sycophancy", "survival-instinct", "refusal", "myopic-reward", "corrigible-neutral-HHH", "coordinate-other-ais"], required=True, help="The dataset to use for testing")
    parser.add_argument("--model_name", type=str, default="gpt2", help="The name of the model to use")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--seed", type=int, default=None, help="The seed to use")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # Print arguments
    print(f"Running with args:")
    print('\n'.join([f'  {k}: {v}' for k, v in vars(args).items()]), end='\n\n')

    # Initialize model
    model = HookedLLM(args.model_name, debug_mode=args.debug)
    
    # Set pad token ID to avoid warnings
    if model.tokenizer.pad_token is None:
        print("Setting pad_token to eos_token to avoid warnings")
        model.tokenizer.pad_token = model.tokenizer.eos_token
    
    # Load steering function
    steering_dir = Path(args.steering_dir)
    
    # Parse the steering config to extract key parameters
    config_parts = args.steering_config.split("_")
    
    # Find the beta and aperture values from the config string
    beta_value = None
    aperture_value = None
    for part in config_parts:
        if part.startswith("beta="):
            beta_value = part.split("=")[1]
        elif part.startswith("aperture="):
            aperture_value = part.split("=")[1]
    
    # Determine whether we're using conceptor or activation steering
    is_conceptor = "conceptor" in args.steering_config
    
    # Determine if using pos_neg and/or additive conceptor
    pos_neg = "pos_neg_conceptors=True" in args.steering_config
    additive = "additive_conceptor=True" in args.steering_config
    
    # Extract dataset name
    dataset_name = None
    for part in config_parts:
        if part.startswith("dataset="):
            dataset_name = part.split("=")[1]
    
    if not dataset_name:
        dataset_name = args.test_dataset
    
    # Construct path to the proper steering function file based on the directory structure
    base_path = steering_dir
    
    # Handle path structure for new saving method
    if is_conceptor:
        # For conceptor-based steering
        base_path = base_path / f"steering_fn_{dataset_name}_conceptor"
        if beta_value:
            base_path = base_path / f"beta={beta_value}"
        if aperture_value:
            base_path = base_path / f"aperture={aperture_value}"
        if additive:
            base_path = base_path / "additive"
        if pos_neg:
            base_path = base_path / "pos_neg"
    else:
        # For activation-based steering
        base_path = base_path / f"steering_fn_{dataset_name}_activation"
        if beta_value:
            base_path = base_path / f"beta={beta_value}"
    
    print(f"Looking for steering function at: {base_path}")
    
    # Check if the safetensors file exists
    safetensors_path = base_path.with_suffix(".safetensors")
    pt_path = base_path.with_suffix(".pt")
    config_path = Path(str(base_path) + "_config.json")
    
    # Verify files exist
    if not safetensors_path.exists() and not pt_path.exists():
        print(f"Steering function file not found at {safetensors_path} or {pt_path}")
        # Try to find any steering function files for debugging
        all_files = []
        for root, _, files in os.walk(steering_dir):
            for file in files:
                if file.endswith(".safetensors") or file.endswith(".pt"):
                    all_files.append(os.path.join(root, file))
        
        if all_files:
            print(f"Available steering functions in directory:")
            for f in sorted(all_files):
                print(f"  - {f}")
        return
    
    if not config_path.exists():
        print(f"Config file not found at {config_path}")
        return
    
    print(f"Loading steering function from: {base_path}")
    
    # Load using appropriate class method
    if is_conceptor:
        steering_fn = ConceptorSteering.load(str(base_path))
        # Move to the correct device
        if device != "cpu":
            for module in steering_fn.conceptors:
                steering_fn.conceptors[module] = steering_fn.conceptors[module].to(device)
    else:
        steering_fn = MeanActivationSteering.load(str(base_path))
        # Move to the correct device
        if device != "cpu":
            for module in steering_fn.vectors:
                steering_fn.vectors[module] = steering_fn.vectors[module].to(device)
    
    if steering_fn is None:
        print("Failed to load steering function. Exiting.")
        return

    # Get test prompts
    print("\n=== LOADING OPEN-ENDED TEST PROMPTS ===")
    # Use the current script directory as the base path
    base_path = Path(__file__).parent
    test_prompts = get_open_ended_test_prompts(args.test_dataset, base_path=base_path)
    
    # Apply max_samples limit if specified
    if args.max_samples is not None and args.max_samples < len(test_prompts):
        test_prompts = test_prompts[:args.max_samples]
        
    print(f"Loaded {len(test_prompts)} test prompts from {args.test_dataset}.")
    
    # Evaluate prompts
    print("\n=== EVALUATING STEERING PERFORMANCE ON OPEN-ENDED QUESTIONS ===")
    results = evaluate_open_ended_with_steering(
        model=model,
        prompts=test_prompts,
        steering_fn=steering_fn,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = base_path / "results" / f"test_open_ended_{args.test_dataset}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save result details
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Save test configuration
    with open(results_dir / "test_config.json", "w") as f:
        json.dump({
            "test_dataset": args.test_dataset,
            "model_name": args.model_name,
            "steering_dir": str(args.steering_dir),
            "steering_config": args.steering_config,
            "max_new_tokens": args.max_new_tokens,
            "timestamp": timestamp
        }, f, indent=4)
    
    print(f"Results saved to {results_dir}")
    print("\nNext steps: You can use an LLM agent to evaluate the quality of the steered vs. baseline outputs.")


if __name__ == "__main__":
    main()
