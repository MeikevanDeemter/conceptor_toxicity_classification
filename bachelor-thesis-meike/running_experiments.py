"""
This run script demonstrates how to use activation steering on simple examples.

Example usage for GPT-J-6B:
```bash
uv run python run_simple_demo.py \
    --examples=wedding \
    --model_name=EleutherAI/gpt-j-6b \
    --layers 15 16 \
    --aperture 0.01 0.1 \
    --beta 0 1 10
```

Example usage for GPT-2:
```bash
uv run python run_simple_demo.py \
    --examples=wedding \
    --model_name=gpt2 \
    --layers 8 9 \
    --aperture 0.01 0.1 \
    --beta 0 1 10
```
"""
# Standard library imports
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
import os

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset

# Local imports
from hookedllm.HookedLLM import HookedLLM
from hookedllm.config import (
    CachingConfig,
    ModuleSelection,
    TokenSelection,
    MeanActivationConfig,
    ConceptorConfig
)
from hookedllm.steering_functions import (
    MeanActivationSteering,
    compute_conceptor,
    ConceptorSteering
)
import create_plots
import mean_activations
import conceptor_classification
import calculate_mean_scores


def get_examples(example_name: str, seed: int = None) -> List[str]:
    # Get the absolute path to the project root directory
    project_root = Path(__file__).parent.parent
    
    # Load the datasets from JSON files
    toxic_dataset = load_dataset(
        "json",
        data_dir=str(project_root / "bachelor-thesis-meike" / "datasets" / f"seed_{seed}"),
        data_files={
            "train": "train_dataset.json",
            "train_toxic": "train_dataset_toxic.json",
            "train_non_toxic": "train_dataset_non_toxic.json",
            "validation": "validation_dataset.json",
            "test": "test_dataset.json"
        }
    )
    
    return {
        "train_dataset": toxic_dataset["train"],
        "train_dataset_toxic": toxic_dataset["train_toxic"],
        "train_dataset_non_toxic": toxic_dataset["train_non_toxic"],
        "validation_dataset": toxic_dataset["validation"],
        "test_dataset": toxic_dataset["test"]
    }[example_name]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="EleutherAI/gpt-neo-1.3B",
        # default="gpt2",
        help="The name of the model to use"
    )
    parser.add_argument(
        "--layer_path",
        type=str,
        nargs='?',
        help="Path to the module list to be steered"
    )
    parser.add_argument(
        "--layers", 
        type=int,
        nargs='*',
        help="The layers in the module list to use"
    )
    parser.add_argument(
        "--layer_modules", 
        type=str, 
        nargs='*',
        help="The layers in the module list to use"
    )   
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode"
    )
    parser.add_argument(
        "--examples",
        type=str,
        choices=["train_toxic_examples", "train_non_toxic_examples", "test_dataset", "test_labels_dataset"],
        default="train_toxic_examples",
        help="The examples to use"
    )
    parser.add_argument(
        "--cache_position",
        type=str,
        default=":1",
        help="* for all tokens, :N for last N tokens"
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
    parser.add_argument(
        "--additive_conceptor",
        action="store_true",
        help="Whether to use additive conceptor steering"
    )
    parser.add_argument(
        "--experiment_type",
        type=str,
        choices=[
                "full_run_model",
                "conceptor_examples", 
                 "conceptor_aperture", 
                 "conceptor_layers", 
                 "conceptor_confusion_matrix",  
                 "mean_activations_layers", 
                 "mean_activations_examples", 
                 "mean_activations_confusion_matrix",
                 "plotting_conceptor_vs_mean_results"],
        default="conceptor_examples",
        help="The type of experiment to run"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        nargs='+',
        default=[3000],
        help="The number of examples to use"
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
        "--use_stored_activations",
        action="store_true",
        help="Whether to use stored activations if they exist"
    )
    parser.add_argument(
        "--seed",
        type=int,
        nargs='+',
        default=[42, 1337, 2024, 7, 8675309, 31415, 12345, 777, 9001, 271828],   
        # default = [42, 1337],
        help="The seed to use"
    )
    
    args = parser.parse_args()

    # default args for tested models
    defaults = {
        'gpt2': {
            'layer_path': 'transformer.h',
            'layers':[2,4,6,8,9,10,11],
            'layer_modules': ['ln_1', 'attn', 'mlp', 'ln_2']
            # 'layers': [1,2,3,4,5,6,7,8,9,10,11],
            # 'layer_modules': ['ln_1', 'attn', 'mlp', 'ln_2']
        },
        'EleutherAI/gpt-neo-125M': {
            'layer_path': 'transformer.h',
            'layers': [2,4,6,8,9,10,11],
            'layer_modules': ['ln_1', 'attn', 'mlp', 'ln_2']
            # 'layers': [11],
            # 'layer_modules': ['ln_2']
        },
        'EleutherAI/gpt-neo-1.3B': {
            'layer_path': 'transformer.h',
            # 'layers': [23],
            # 'layer_modules': ['ln_2']
            'layers': [2,6,10,14,18,19,20,21,22,23],	
            'layer_modules': ['ln_1', 'attn', 'mlp', 'ln_2']
        },
        'EleutherAI/gpt-neo-2.7B': {
            'layer_path': 'transformer.h',
            'layers': [2,6,10,14,18,22,26,27,28,29,30,31],
            'layer_modules': ['ln_1', 'attn', 'mlp', 'ln_2']
        }
    }
    # Ensure module specification
    for arg in ['layer_path', 'layers', 'layer_modules']: 
        if not getattr(args, arg):
            setattr(args, arg, defaults[args.model_name][arg])

    # Parse cache position
    if args.cache_position == "*":
        args.cache_position = None
    elif args.cache_position.startswith(":"):
        args.cache_position = list(range(-int(args.cache_position[1:]), 0))
    else:
        raise ValueError(f"Invalid cache position: {args.cache_position}")

    return args


def print_args(args):
    print(f"Running with args:")
    print('\n'.join([f'  {k}: {v}' for k, v in vars(args).items()]), end='\n\n')


def calculate_activations(model, caching_config, examples, type):
    """
    Calculate activations for a list of examples.
    """
    with model.caching(caching_config):
        for idx, example in enumerate(examples):
            print(f"Calculating activations for {type} example {idx} of {len(examples)}")
            model.forward(example)
            # Clear GPU cache periodically to prevent memory buildup
            if idx % 100 == 0 and idx > 0:
                torch.cuda.empty_cache()
    activations = model.get_and_clear_cache()
    activations = {
        k: torch.cat([t.cpu() for t in v], dim=0)
        for k, v in activations.items()
    }
    
    return activations


def sanitize_model_name(args, model_name):
    """Convert model name to a safe filename by replacing slashes with hyphens."""
    args.model_name = model_name.replace('/', '-').replace('\\', '-')
    return args.model_name


def get_activation_path(activations_name):
    """Get the path where activations should be stored."""
    return os.path.join("stored_activations", activations_name)


def load_activations(activations_name):
    """Load activations from disk if they exist."""
    activation_path = get_activation_path(activations_name)
    if os.path.exists(activation_path):
        print(f"Loading stored activations from {activation_path}")
        return torch.load(activation_path)
    return None


def store_activations(activations, storage_name):
    """Store activations to disk if they don't exist yet."""
    activation_path = get_activation_path(storage_name)
    if not os.path.exists(activation_path):
        os.makedirs(os.path.dirname(activation_path), exist_ok=True)
        print(f"Storing activations to {activation_path}")
        torch.save(activations, activation_path)
    else:
        print(f"Activations already exist at {activation_path}")

def setup_experiments(args, seed, model):
    """Setup experiments for a specific seed."""

    module_selection = ModuleSelection(
        layers=args.layers,
        layer_modules=args.layer_modules,
        layer_path=args.layer_path,
    )
    
    # Sanitize model name for file operations
    model_name = sanitize_model_name(args, args.model_name)


    # Extract activations for a specific concept
    caching_config = CachingConfig(
        modules=module_selection,
        tokens=TokenSelection(position=args.cache_position),
    )
    
    # Get all datasets first
    print(f"\nLoading datasets for seed {seed}...")
    train_full_dataset = get_examples("train_dataset", seed)
    train_dataset_toxic = get_examples("train_dataset_toxic", seed)
    train_dataset_non_toxic = get_examples("train_dataset_non_toxic", seed)
    validation_dataset = get_examples("validation_dataset", seed)
    test_dataset = get_examples("test_dataset", seed)
    

    # Try to load stored activations if requested
    if args.use_stored_activations:
        activations_name = f"{model_name}_seed_{seed}_activations.pt"
        stored_activations = load_activations(activations_name)
        if stored_activations is not None:
            train_activations = stored_activations['train']
            validation_set_activations = stored_activations['validation']
            test_set_activations = stored_activations['test']
            print("Using stored activations")
            #print all keys of train_activations
        else:
            print("No stored activations found, computing new ones...")
            args.use_stored_activations = False

    # Compute activations if not using stored ones
    if not args.use_stored_activations:
        print("\nComputing activations for toxic training data, non-toxic training data and test data...")
        train_activations = calculate_activations(model, caching_config, train_full_dataset["text"], "full training set")
        validation_set_activations = calculate_activations(model, caching_config, validation_dataset["text"], "validation set")
        test_set_activations = calculate_activations(model, caching_config, test_dataset["text"], "test set")

        # Store the computed activations
        activations_to_store = {
            'train': train_activations,
            'validation': validation_set_activations,
            'test': test_set_activations
        }
        storage_name = f"{model_name}_seed_{seed}_activations.pt"
        store_activations(activations_to_store, storage_name)

    # Split the train_activations into toxic and non-toxic
    train_toxic_activations = {k: v[:len(train_dataset_toxic)] for k, v in train_activations.items()}
    train_non_toxic_activations = {k: v[len(train_dataset_toxic):] for k, v in train_activations.items()}

    # Get all module names from the activations dictionary
    module_names = list(train_toxic_activations.keys())
    print(f"\nExtracted activations for modules: {module_names}")
    # Initialize scores dictionary
    scores = {
        'test_set_score': {}, 
        'validation_set_score': {},
        'training_set_score': {}
    }
    
    return args, scores, train_toxic_activations, train_non_toxic_activations, validation_set_activations, validation_dataset, test_set_activations, test_dataset

def initialize_scores_dictionary():
    """Create a fresh scores dictionary for each experiment."""
    scores = {
        'test_set_score': {}, 
        'validation_set_score': {},
        'training_set_score': {}
    }
    return scores


def main():
    args = parse_args()
    print_args(args)
    
    if args.experiment_type == "full_run_model":
        print(f"\nRunning full run for model {args.model_name}...")
        
        # Load model once for all seeds
        model = HookedLLM(args.model_name, debug_mode=args.debug)
        
        for seed in args.seed:
            print(f"\nRunning full run for seed {seed}...")
            args, _, train_toxic_activations, train_non_toxic_activations, validation_set_activations, validation_dataset, test_set_activations, test_dataset = setup_experiments(args, seed, model)

            # Create fresh scores dictionary for each experiment

            
            print(f"\nRunning layers experiment for seed {seed}...")
            args.experiment_type = "conceptor_layers"
            scores = initialize_scores_dictionary()
            conceptor_classification.layers(args, 
                                   scores,
                                   train_toxic_activations, 
                                   train_non_toxic_activations, 
                                   test_set_activations,
                                   test_dataset,
                                   seed)
            
            print(f"\nRunning aperture experiment for seed {seed}...")
            args.experiment_type = "conceptor_aperture"
            scores = initialize_scores_dictionary()
            conceptor_classification.aperture(args, 
                                   scores,
                                   train_toxic_activations, 
                                   train_non_toxic_activations, 
                                   validation_set_activations, 
                                   validation_dataset,
                                   seed)
            
            print(f"\nRunning examples experiment for seed {seed}...")
            args.experiment_type = "conceptor_examples"
            scores = initialize_scores_dictionary()
            conceptor_classification.examples(args, 
                                   scores,
                                   train_toxic_activations, 
                                   train_non_toxic_activations, 
                                   test_set_activations,
                                   test_dataset,
                                   seed)
            
            print(f"\nRunning mean activations layers experiment for seed {seed}...")
            args.experiment_type = "mean_activations_layers"
            scores = initialize_scores_dictionary()
            mean_activations.layers(args, 
                                   scores,
                                   train_toxic_activations, 
                                   train_non_toxic_activations, 
                                   test_set_activations,
                                   test_dataset,
                                   seed)
            
            print(f"\nRunning mean activations examples experiment for seed {seed}...")
            args.experiment_type = "mean_activations_examples"
            scores = initialize_scores_dictionary()
            mean_activations.examples(args, 
                                   scores,
                                   train_toxic_activations, 
                                   train_non_toxic_activations, 
                                   test_set_activations,
                                   test_dataset,
                                   seed)
            
        # # Calculate mean scores for both approaches
        # print("Processing conceptor-based and mean activation-based results...")
        # calculate_mean_scores.main()
            
    
 
    
    # Calculate results for each experiment type
    elif args.experiment_type == "conceptor_examples":
        conceptor_classification.examples(args, 
                                   scores,
                                   train_toxic_activations, 
                                   train_non_toxic_activations, 
                                   test_set_activations,
                                   test_dataset)
    
    elif args.experiment_type == "conceptor_aperture":
        args, 
        scores, 
        train_toxic_activations, 
        train_non_toxic_activations, 
        validation_set_activations, 
        validation_dataset,
        test_set_activations, 
        test_dataset = setup_experiments(args, seed, model)
        
        conceptor_classification.aperture(args, 
                                   scores, 
                                   train_toxic_activations, 
                                   train_non_toxic_activations, 
                                   validation_set_activations, 
                                   validation_dataset)
        
    elif args.experiment_type == "conceptor_layers":
        conceptor_classification.layers(args, 
                                 scores, 
                                 train_toxic_activations, 
                                 train_non_toxic_activations, 
                                 validation_set_activations, 
                                 validation_dataset)

    elif args.experiment_type == "conceptor_confusion_matrix":
        conceptor_classification.confusion_matrix(args, 
                                                train_toxic_activations, 
                                                train_non_toxic_activations, 
                                                test_set_activations, 
                                                test_dataset,
                                                seed)
    
    elif args.experiment_type == "mean_activations_layers":
        mean_activations.layers(args, 
                                scores,
                                train_toxic_activations, 
                                train_non_toxic_activations, 
                                validation_set_activations, 
                                validation_dataset)
    
    elif args.experiment_type == "mean_activations_examples":
        mean_activations.examples(args, 
                                scores,
                                train_toxic_activations, 
                                train_non_toxic_activations, 
                                test_set_activations, 
                                test_dataset)
        
    elif args.experiment_type == "mean_activations_confusion_matrix":
        mean_activations.confusion_matrix(args, 
                                          train_toxic_activations, 
                                          train_non_toxic_activations, 
                                          test_set_activations, 
                                          test_dataset)
        
    elif args.experiment_type == "plotting_conceptor_vs_mean_results":
        create_plots.plotting_stored_results(args.seed)

    else:
        print(f"Experiment type {args.experiment_type} not found")


if __name__ == "__main__":
    main()
