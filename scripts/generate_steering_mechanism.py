"""
Generate steering functions for complex behaviors in language models.

This script generates steering functions from a dataset of examples and saves them for later use.
Below an example usage of this script is given.
You can find the results in the ./results/ directory.
Here, it will be called {steering_function}_{timestamp}/
Then, you can use the @ab_test_steering.py script to test the steering functions on a test set.
This can be done by following the instructions in the ab_test_steering.py script.

Example usage:
```bash
python scripts/generate_steering_mechanism.py \
    --dataset=hallucination \
    --model_name=gpt2 \
    --layers 8 \
    --aperture 1 \
    --beta 3.85 \
    --steering_function=conceptor \
    --additive_conceptor \
    --pos_neg_conceptors
```
"""

import argparse
import json
import random
import os
from datetime import datetime
from typing import List, Dict, Any
import itertools
from pathlib import Path
import torch
import sys

# Add the parent directory to the path to ensure our local modules are found first
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the local create_dataset_splits function
from datasets.create_validation_split import create_dataset_splits

from hookedllm.HookedLLM import HookedLLM
from hookedllm.config import CachingConfig, ModuleSelection, TokenSelection
from hookedllm.config import ConceptorConfig, MeanActivationConfig
from hookedllm.steering_functions import ConceptorSteering, MeanActivationSteering
from hookedllm.utils import (
    get_examples,
    collect_activations,
    create_steering_function
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2", help="The name of the model to use")
    parser.add_argument("--layers", type=int, nargs='+', default=[6], help="The layers to use")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--seed", type=int, default=None, help="The seed to use")
    parser.add_argument("--dataset", type=str, choices=["hallucination", "sycophancy", "survival-instinct", "refusal", "myopic-reward", "corrigible-neutral-HHH", "coordinate-other-ais"], default="hallucination", help="The dataset to use for steering function generation")
    parser.add_argument("--cache_position", type=str, default="-2", help="* for all tokens, :N for last N tokens, -2 for the A/B token specifically")
    parser.add_argument("--beta", type=float, nargs='+', default=[0.0, 0.5, 1.0, 3.0, 5.0, 10.0], help="The beta values to use")
    parser.add_argument("--steering_function", type=str, choices=["activation", "conceptor"], default="conceptor", help="The steering function to use")
    parser.add_argument("--max_examples", type=int, default=50, help="Maximum number of examples to use")
    # conceptor-specific args
    parser.add_argument("--aperture", type=float, nargs='+', default=[0.01], help="The aperture values to use")
    parser.add_argument("--additive_conceptor", action="store_true", help="Whether to use additive conceptor steering")
    parser.add_argument("--pos_neg_conceptors", action="store_true", help="Whether to compute C_pos AND (NOT C_neg) for conceptor steering.")
    parser.add_argument("--skip_dataset_creation", action="store_true", help="Skip dataset creation step (useful if datasets are already created)")
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
    """Print command line arguments in a readable format."""
    print(f"Running with args:")
    print('\n'.join([f'  {k}: {v}' for k, v in vars(args).items()]), end='\n\n')


def ensure_datasets_exist(dataset_name=None, verbose=True):
    """
    Ensure datasets exist without recreating them if they're already available.
    
    Args:
        dataset_name: Optional specific dataset to create. If None, checks all datasets.
        verbose: Whether to print detailed progress information.
    
    Returns:
        List of successfully created datasets.
    """
    print("\n=== CHECKING FOR DATASETS ===")
    
    # Check first if the dataset already exists
    if dataset_name:
        dataset_path = Path(os.path.join('datasets', 'processed', dataset_name))
        if dataset_path.exists() and any(dataset_path.iterdir()):
            print(f"Dataset {dataset_name} already exists, skipping creation.")
            return [dataset_name]
    
    # If we reach here, we need to create the datasets
    print("Creating missing datasets...")
    successful_behaviors = create_dataset_splits(verbose=verbose)
    print(f"Successfully created datasets for: {', '.join(successful_behaviors)}")
    
    if dataset_name and dataset_name not in successful_behaviors:
        print(f"WARNING: The requested dataset '{dataset_name}' was not successfully created.")
    
    return successful_behaviors


def main():
    args = parse_args()

    # Set random seeds for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # Print args nicely
    print_args(args)
    
    # Ensure datasets exist by creating them from raw data
    if not args.skip_dataset_creation:
        ensure_datasets_exist(dataset_name=args.dataset)
    else:
        print("Skipping dataset creation step as requested.")

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
    generate_matching_examples = get_examples(args.dataset, generate=True, match=True)
    generate_not_matching_examples = get_examples(args.dataset, generate=True, match=False)
    
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
    caching_config = CachingConfig(
        modules=module_selection,
        tokens=TokenSelection(position=args.cache_position),
    )
    
    # Collect activations for training the steering function
    print("\n=== COLLECTING ACTIVATIONS FOR STEERING ===")
    max_examples = args.max_examples
    
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
    
    # Create and save steering functions for each hyperparameter combination
    print("\n=== GENERATING STEERING FUNCTIONS ===")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) / "results" / f"{args.dataset}_{args.steering_function}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save general configuration
    with open(results_dir / "general_config.json", "w") as f:
        json.dump({
            "dataset": args.dataset,
            "model_name": args.model_name,
            "layers": args.layers,
            "cache_position": args.cache_position,
            "steering_function": args.steering_function,
            "timestamp": timestamp
        }, f, indent=4)
    
    # Create and save steering functions for each hyperparameter combination
    for hyperparam_tuple in hyperparam_combinations:
        # Map hyperparameter names to values
        hyperparam_dict = dict(zip(hyperparam_names, hyperparam_tuple))
        hyperparam_strl = [f"{k}={v}" for k, v in hyperparam_dict.items()]
        print(f"\n----- Generating with {', '.join(hyperparam_strl)} -----")

        # Create the appropriate steering function using the utility function
        steering_fn = create_steering_function(
            steering_function_type=args.steering_function,
            module_selection=module_selection,
            positive_activations=not_matching_act,  # Good behavior activation (not_matching)
            negative_activations=matching_act,      # Bad behavior activation (matching)
            hyperparam_dict=hyperparam_dict,
            use_additive_conceptor=args.additive_conceptor,
            use_pos_neg_conceptors=args.pos_neg_conceptors
        )
        
        # Create save path with appropriate naming
        save_path = results_dir / f"steering_fn_{args.dataset}_{args.steering_function}"
        for k, v in hyperparam_dict.items():
            save_path = save_path / f"{k}={v}"
        if args.steering_function == "conceptor" and args.additive_conceptor:
            save_path = save_path / "additive"
        if args.steering_function == "conceptor" and args.pos_neg_conceptors:
            save_path = save_path / "pos_neg"
            
        # Create the directory if it doesn't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the steering function using its built-in save method
        steering_fn.save(str(save_path))
        
        # Also save a config file with human-readable information
        config_info = {
            "dataset": args.dataset,
            "model_name": args.model_name,
            "layers": args.layers,
            "steering_function": args.steering_function,
            "additive_conceptor": args.additive_conceptor if args.steering_function == "conceptor" else None,
            "pos_neg_conceptors": args.pos_neg_conceptors if args.steering_function == "conceptor" else None,
            **hyperparam_dict
        }
        with open(save_path.parent / f"{save_path.name}_info.json", "w") as f:
            json.dump(config_info, f, indent=4)
    
    print(f"\nAll steering functions saved to {results_dir}")


if __name__ == "__main__":
    main() 