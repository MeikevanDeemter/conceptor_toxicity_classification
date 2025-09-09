"""
Test steering functions on different datasets.

This script loads a pre-generated steering function and applies it to a test dataset.
The steering function is saved in the ./results/ directory.
You have to specify the {steering_function}_{timestamp}/ directory where the steering function is saved (through @generate_steering.py)
You have to specify the steering_config which is the config used to generate the steering function (through @generate_steering.py)
To do this, you can copy the relative path of the config, and paste it in. An example is given below, which should run (after you have run @generate_steering.py).
Example usage:
```bash
python scripts/ab_test_steering.py \
  --steering_dir="./results/hallucination_conceptor_20250430_171006" \
  --steering_config="dataset=hallucination_model_name=gpt2_layers=[8]_steering_function=conceptor_additive_conceptor=True_pos_neg_conceptors=True_beta=3.85_aperture=1.0" \
  --test_dataset=hallucination \
  --model_name=gpt2
```
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import torch
import sys

# Add the parent directory to the path to ensure our local modules are found first
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the local create_dataset_splits function
from datasets.create_validation_split import create_dataset_splits

from hookedllm.HookedLLM import HookedLLM
from hookedllm.steering_functions import ConceptorSteering, MeanActivationSteering, SteeringFunction
from hookedllm.utils import (
    get_test_prompts,
    evaluate_prompts_with_logits,
    calculate_accuracy,
    BLUE, RESET
)


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steering_dir", type=str, required=True, help="Directory containing the saved steering function")
    parser.add_argument("--steering_config", type=str, required=True, help="Configuration pattern to match for loading the steering function (e.g., 'beta=3.85_aperture=0.01')")
    parser.add_argument("--test_dataset", type=str, choices=["hallucination", "sycophancy", "survival-instinct", "refusal", "myopic-reward", "corrigible-neutral-HHH", "coordinate-other-ais"], required=True, help="The dataset to use for testing")
    parser.add_argument("--model_name", type=str, default="gpt2", help="The name of the model to use")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--seed", type=int, default=None, help="The seed to use")
    parser.add_argument("--skip_dataset_creation", action="store_true", help="Skip dataset creation step (useful if datasets are already created)")
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

    # Ensure datasets exist by creating them from raw data
    if not args.skip_dataset_creation:
        ensure_datasets_exist(dataset_name=args.test_dataset)
    else:
        print("Skipping dataset creation step as requested.")
    
    # Initialize model
    model = HookedLLM(args.model_name, debug_mode=args.debug)
    
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
    
    # Construct path to the proper steering function file based on the directory structure
    base_path = steering_dir
    
    # Handle path structure for new saving method
    if is_conceptor:
        # For conceptor-based steering
        base_path = base_path / f"steering_fn_hallucination_conceptor"
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
        base_path = base_path / f"steering_fn_hallucination_activation"
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
    print("\n=== LOADING TEST PROMPTS ===")
    test_prompts = get_test_prompts(args.test_dataset)
    print(f"Loaded {len(test_prompts)} test prompts from {args.test_dataset}.")
    
    # Evaluate prompts
    print("\n=== EVALUATING STEERING PERFORMANCE ===")
    results = evaluate_prompts_with_logits(
        model=model,
        prompts=test_prompts,
        steering_fn=steering_fn
    )
    
    # Calculate accuracy
    accuracy_stats = calculate_accuracy(results)
    
    # Print accuracy results
    print("\n=== ACCURACY RESULTS ===")
    good_pct = accuracy_stats['matches_good_behavior'] / accuracy_stats['total'] * 100 if accuracy_stats['total'] > 0 else 0
    bad_pct = accuracy_stats['matches_bad_behavior'] / accuracy_stats['total'] * 100 if accuracy_stats['total'] > 0 else 0
    amb_pct = accuracy_stats['ambiguous'] / accuracy_stats['total'] * 100 if accuracy_stats['total'] > 0 else 0
    print(f"Steering function: {args.steering_config}")
    print(f"  - Matches good behavior: {good_pct:.2f}% ({accuracy_stats['matches_good_behavior']}/{accuracy_stats['total']})")
    print(f"  - Matches bad behavior: {bad_pct:.2f}% ({accuracy_stats['matches_bad_behavior']}/{accuracy_stats['total']})")
    print(f"  - Ambiguous/Other: {amb_pct:.2f}% ({accuracy_stats['ambiguous']}/{accuracy_stats['total']})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parents[0] / "results" / f"test_{args.test_dataset}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save result details
    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Save accuracy summary
    with open(results_dir / "accuracy.json", "w") as f:
        json.dump(accuracy_stats, f, indent=4)
    
    # Save test configuration
    with open(results_dir / "test_config.json", "w") as f:
        json.dump({
            "test_dataset": args.test_dataset,
            "model_name": args.model_name,
            "steering_dir": str(args.steering_dir),
            "steering_config": args.steering_config,
            "timestamp": timestamp
        }, f, indent=4)
    
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main() 