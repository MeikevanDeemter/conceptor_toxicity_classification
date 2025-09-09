"""
This run script demonstrates how to use activation steering on complex behaviors. 

Example usage for GPT-J-6B:
```bash
uv run python complex_behavior.py \
    --generate_dataset=hallucination \
    --test_dataset=hallucination \
    --model_name=EleutherAI/gpt-j-6b \
    --layers 15 16 \
    --aperture 0.01 0.1 \
    --beta 0 1 10 \
    --additive_conceptor \ 
    --pos_neg_conceptors
```

Example usage for GPT-2XL: (gpt2 is likely too small)
```bash
uv run python complex_behavior.py \
    --generate_dataset=hallucination \
    --test_dataset=hallucination \
    --model_name=gpt2-xl \
    --layers 16 \
    --aperture 0.01 0.1 0.04956085950553879 \
    --beta 0 1 10 3.850478575207825\
    --additive_conceptor \
    --pos_neg_conceptors
```
Got best hyperparameters from Joris' run to do quick testing
python complex_behavior.py --generate_dataset hallucination --test_dataset hallucination --cache_position -2 --steering_function activation --model_name gpt2 --layers 8 --beta 3.850478575207825 0
python complex_behavior.py --generate_dataset hallucination --test_dataset hallucination --cache_position -2 --steering_function conceptor --model_name gpt2 --layers 8 --aperture 0.04956085950553879 --beta 3.850478575207825

"""

#potential TODO: (optional but could speedup), Steven has a function that allows you to compute a conceptor once and adjust to new apertures without having to completely recompute conceptor. 

import argparse
import json
import re
import random
import os
from functools import partial
from datetime import datetime
from typing import List, Dict, Any, Tuple
import itertools
from pathlib import Path
import torch

from hookedllm.HookedLLM import HookedLLM
from hookedllm.config import CachingConfig, ModuleSelection, TokenSelection
from hookedllm.config import MeanActivationConfig
from hookedllm.steering_functions import MeanActivationSteering
from hookedllm.config import ConceptorConfig
from hookedllm.steering_functions import ConceptorSteering, SteeringFunction
from hookedllm.utils import (
    get_examples, 
    get_test_prompts, 
    collect_activations, 
    create_steering_function, 
    evaluate_prompts_with_logits, 
    calculate_accuracy,
    BOLD, RESET, BLUE
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="gpt2",
        help="The name of the model to use"
    )
    parser.add_argument(
        "--layers", 
        type=int, 
        nargs='+',
        default=[6],
        help="The layers to use"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The seed to use"
    )
    parser.add_argument(
        "--generate_dataset",
        type=str,
        choices=["hallucination", "sycophancy", "survival-instinct", "refusal", "myopic-reward", "corrigible-neutral-HHH", "coordinate-other-ais"],
        default="hallucination",
        help="The dataset to use for steering function generation"
    )
    parser.add_argument(
        "--test_dataset",
        type=str,
        choices=["hallucination", "sycophancy", "survival-instinct", "refusal", "myopic-reward", "corrigible-neutral-HHH", "coordinate-other-ais"],
        default="hallucination",
        help="The dataset to use for steering function testing"
    )
    parser.add_argument(
        "--cache_position",
        type=str,
        default="-2", # so that it captures the A/B token specifically and not the ")" token.
        help="* for all tokens, :N for last N tokens, -2 for the A/B token specifically"
    )
    parser.add_argument(
        "--beta",
        type=float,
        nargs='+',
        default=[0.0, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0],
        help="The beta values to use"
    )
    parser.add_argument(
        "--steering_function",
        type=str,
        choices=["activation", "conceptor"],
        default="conceptor",
        help="The steering function to use"
    )
    # conceptor-specific args
    parser.add_argument(
        "--aperture",
        type=float,
        nargs='+',
        default=[0.01],
        help="The aperture values to use"
    )
    parser.add_argument(
        "--additive_conceptor",
        action="store_true",
        help="Whether to use additive conceptor steering"
    )
    parser.add_argument(
        "--pos_neg_conceptors",
        action="store_true",
        help="Whether to compute C_pos AND (NOT C_neg) for conceptor steering."
    )
    args = parser.parse_args()

    # Parse cache position
    if args.cache_position == "*":
        args.cache_position = None
    elif args.cache_position.startswith(":"):
        args.cache_position = list(range(-int(args.cache_position[1:]), 0))
    elif args.cache_position == "-2":
        # Special case: target the A/B token (second-to-last token)
        args.cache_position = [-2]
    else:
        raise ValueError(f"Invalid cache position: {args.cache_position}")

    return args


def print_args(args):
    print(f"Running with args:")
    print('\n'.join([f'  {k}: {v}' for k, v in vars(args).items()]), end='\n\n')


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Print args nicely
    print_args(args)

    # Initialize model and module selection
    model = HookedLLM(args.model_name, debug_mode=args.debug)
    module_selection = ModuleSelection(
        layers=args.layers,
        layer_modules=["ln_1"],
        layer_path="transformer.h",
    )

    # Get dataset examples for training the steering function
    print("\n=== LOADING TRAINING EXAMPLES ===")
    print("For training, we need both good and bad behavior examples:")
    generate_matching_examples = get_examples(args.generate_dataset, generate=True, match=True)
    generate_not_matching_examples = get_examples(args.generate_dataset, generate=True, match=False)
    
    # Print a few examples to check
    print("\nGood behavior examples (first 2):")
    for i in range(min(2, len(generate_not_matching_examples))):
        example = generate_not_matching_examples[i]
        print(f"Example {i+1}: {example}")
        print(f"Answer: {example.strip()[-3:] if len(example.strip()) >= 3 else 'Unknown'}")
    
    print("\nBad behavior examples (first 2):")
    for i in range(min(2, len(generate_matching_examples))):
        example = generate_matching_examples[i]
        print(f"Example {i+1}: {example}")
        print(f"Answer: {example.strip()[-3:] if len(example.strip()) >= 3 else 'Unknown'}")
    
    # Define caching strategy based on command line args
    # Example: --cache_position -2 targets the second-to-last token (often the A/B choice)
    caching_config = CachingConfig(
        modules=module_selection,
        tokens=TokenSelection(position=args.cache_position),
    )

    # Get test prompts to evaluate the model
    print("\n=== LOADING TEST PROMPTS ===")
    test_prompts = get_test_prompts(args.test_dataset)
    
    # Print test prompts for verification
    print("\nTest prompts (first 2):")
    for i in range(min(2, len(test_prompts))):
        prompt_data = test_prompts[i]
        print(f"Prompt {i+1}: {prompt_data['question']}")
        print(f"Good behavior answer: {prompt_data['good_behavior_answer']}")
        print(f"Bad behavior answer: {prompt_data['bad_behavior_answer']}")
    
    # Collect activations for training the steering function
    print("\n=== COLLECTING ACTIVATIONS FOR STEERING ===")
    max_examples = 50  # TODO: Process all examples or increase this limit, now just testing
    
    # We need activations from BOTH good and bad behavior examples to create the steering vector
    print("Collecting activations from bad behavior examples...")
    matching_act = collect_activations(model, generate_matching_examples, caching_config, max_examples)
    print(f"Collected activations for {min(max_examples, len(generate_matching_examples))} bad behavior examples.")
    
    print("Collecting activations from good behavior examples...")
    not_matching_act = collect_activations(model, generate_not_matching_examples, caching_config, max_examples)
    print(f"Collected activations for {min(max_examples, len(generate_not_matching_examples))} good behavior examples.")

    # Setup hyperparameters based on steering function type
    if args.steering_function == "activation":
        hyperparam_names = ["beta"]
        hyperparam_values = [args.beta]
    elif args.steering_function == "conceptor":
        hyperparam_names = ["beta", "aperture"]
        hyperparam_values = [args.beta, args.aperture]
    else:
        raise ValueError(f"Invalid steering function: {args.steering_function}")

    # Generate all combinations of hyperparameters
    hyperparam_combinations = list(itertools.product(*hyperparam_values))
    all_results = {}
    
    # Test each combination of hyperparameters
    print("\n=== EVALUATING STEERING PERFORMANCE ===")
    for hyperparam_tuple in hyperparam_combinations:
        # Map hyperparameter names to values
        hyperparam_dict = dict(zip(hyperparam_names, hyperparam_tuple))
        hyperparam_strl = [f"{k}={v}" for k, v in hyperparam_dict.items()]
        print(f"\n----- Running with {', '.join(hyperparam_strl)} -----")

        # Create the appropriate steering function
        steering_fn = create_steering_function(
            steering_function_type=args.steering_function,
            module_selection=module_selection,
            positive_activations=not_matching_act,  # Good behavior activation (not_matching)
            negative_activations=matching_act,      # Bad behavior activation (matching)
            hyperparam_dict=hyperparam_dict,
            use_additive_conceptor=args.additive_conceptor,
            use_pos_neg_conceptors=args.pos_neg_conceptors
        )
        
        # Evaluate prompts using the steering function and record results
        results = evaluate_prompts_with_logits(
            model=model,
            prompts=test_prompts,
            steering_fn=steering_fn,
            hyperparam_strl=hyperparam_strl
        )
        
        # Add to all results
        all_results.update(results)

    # Calculate accuracy statistics
    accuracy_stats = calculate_accuracy(all_results)

    # Print overall accuracy for each hyperparameter configuration
    print("\n===== ACCURACY RESULTS =====")
    for config, stats in accuracy_stats.items():
        good_pct = stats['matches_good_behavior'] / stats['total'] * 100 if stats['total'] > 0 else 0
        bad_pct = stats['matches_bad_behavior'] / stats['total'] * 100 if stats['total'] > 0 else 0
        amb_pct = stats['ambiguous'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"Configuration {config}:")
        print(f"  - Matches good behavior: {good_pct:.2f}% ({stats['matches_good_behavior']}/{stats['total']})")
        print(f"  - Matches bad behavior: {bad_pct:.2f}% ({stats['matches_bad_behavior']}/{stats['total']})")
        print(f"  - Ambiguous/Other: {amb_pct:.2f}% ({stats['ambiguous']}/{stats['total']})")
    
    # Find the best hyperparameter configuration (maximizing good behavior)
    if accuracy_stats:
        best_config = max(accuracy_stats.items(), key=lambda x: x[1]['matches_good_behavior'])
        print(f"\nBest configuration: {best_config[0]}")
        print(f"  - Matches good behavior: {best_config[1]['matches_good_behavior']/best_config[1]['total']*100:.2f}%")
        print(f"  - Matches bad behavior: {best_config[1]['matches_bad_behavior']/best_config[1]['total']*100:.2f}%")
        print(f"  - Hyperparameters: {best_config[0]}")

    # Save results to results folder directory in the root of repository
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parents[1] / "results" / f"results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    with open(results_dir / "accuracy.json", "w") as f:
        json.dump(accuracy_stats, f, indent=4)
    with open(results_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()
