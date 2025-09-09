#!/usr/bin/env python3

import json
import os
import numpy as np
from collections import defaultdict

def extract_experiment_data():
    """Extract all experiment data across all seeds and models."""
    results_dir = "results"
    models = ["EleutherAI-gpt-neo-125M", "EleutherAI-gpt-neo-1.3B", "EleutherAI-gpt-neo-2.7B"]
    
    # Data structure: {model: {experiment_type: {n_examples: [scores]}}}
    all_data = {}
    
    for model in models:
        all_data[model] = {
            'conceptor_examples': defaultdict(list),
            'mean_activations_examples': defaultdict(list),
            'conceptor_layers': defaultdict(list),
            'mean_activations_layers': defaultdict(list)
        }
    
    # Get all seed directories
    seed_dirs = [d for d in os.listdir(results_dir) if d.startswith('seed_') and os.path.isdir(os.path.join(results_dir, d))]
    
    print(f"Analyzing experiments from {len(seed_dirs)} seed directories...")
    
    for seed_dir in seed_dirs:
        seed_path = os.path.join(results_dir, seed_dir)
        
        for model in models:
            # Check conceptor examples
            conceptor_path = os.path.join(seed_path, "conceptor_examples", model, "raw_results")
            if os.path.exists(conceptor_path):
                for file_name in os.listdir(conceptor_path):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(conceptor_path, file_name)
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            
                            test_scores = data.get('metrics', {}).get('test_scores', {})
                            for n_examples_str, accuracy in test_scores.items():
                                n_examples = int(n_examples_str)
                                all_data[model]['conceptor_examples'][n_examples].append(accuracy)
                        except Exception as e:
                            print(f"Error reading conceptor file {file_path}: {e}")
            
            # Check mean activations examples
            mean_activations_path = os.path.join(seed_path, "mean_activations_examples", model, "raw_results")
            if os.path.exists(mean_activations_path):
                for file_name in os.listdir(mean_activations_path):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(mean_activations_path, file_name)
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            
                            test_scores = data.get('metrics', {}).get('test_scores', {})
                            for n_examples_str, accuracy in test_scores.items():
                                n_examples = int(n_examples_str)
                                all_data[model]['mean_activations_examples'][n_examples].append(accuracy)
                        except Exception as e:
                            print(f"Error reading mean activations file {file_path}: {e}")
            
            # Check conceptor layers
            conceptor_layers_path = os.path.join(seed_path, "conceptor_layers", model, "raw_results")
            if os.path.exists(conceptor_layers_path):
                for file_name in os.listdir(conceptor_layers_path):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(conceptor_layers_path, file_name)
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            
                            # Use validation_scores and layer names as keys
                            val_scores = data.get('metrics', {}).get('validation_scores', {})
                            for layer_name, accuracy in val_scores.items():
                                all_data[model]['conceptor_layers'][layer_name].append(accuracy)
                        except Exception as e:
                            print(f"Error reading conceptor layers file {file_path}: {e}")
            
            # Check mean activations layers
            mean_activations_layers_path = os.path.join(seed_path, "mean_activations_layers", model, "raw_results")
            if os.path.exists(mean_activations_layers_path):
                for file_name in os.listdir(mean_activations_layers_path):
                    if file_name.endswith('.json'):
                        file_path = os.path.join(mean_activations_layers_path, file_name)
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                            
                            # Use validation_scores and layer names as keys
                            val_scores = data.get('metrics', {}).get('validation_scores', {})
                            for layer_name, accuracy in val_scores.items():
                                all_data[model]['mean_activations_layers'][layer_name].append(accuracy)
                        except Exception as e:
                            print(f"Error reading mean activations layers file {file_path}: {e}")
    
    return all_data

def calculate_mean_scores(all_data):
    """Calculate mean scores across all seeds for each experiment type."""
    results = {}
    
    for model, experiment_types in all_data.items():
        results[model] = {}
        
        for experiment_type, data in experiment_types.items():
            experiment_stats = {}
            
            # Calculate mean for each parameter value
            for param_value, scores in data.items():
                if scores:  # Only if we have data
                    # If scores are dicts, extract metrics; else use as is
                    if isinstance(scores[0], dict):
                        accs = [s['accuracy'] for s in scores if isinstance(s, dict) and 'accuracy' in s]
                        precs = [s['precision'] for s in scores if isinstance(s, dict) and 'precision' in s]
                        recs = [s['recall'] for s in scores if isinstance(s, dict) and 'recall' in s]
                        f1s = [s['f1_score'] for s in scores if isinstance(s, dict) and 'f1_score' in s]
                    else:
                        accs = scores
                        precs = recs = f1s = []
                    mean_score = np.mean(accs)
                    std_score = np.std(accs)
                    mean_prec = np.mean(precs) if precs else None
                    std_prec = np.std(precs) if precs else None
                    mean_rec = np.mean(recs) if recs else None
                    std_rec = np.std(recs) if recs else None
                    mean_f1 = np.mean(f1s) if f1s else None
                    std_f1 = np.std(f1s) if f1s else None
                    count = len(accs)
                    
                    experiment_stats[param_value] = {
                        'mean': float(mean_score),
                        'std': float(std_score),
                        'mean_precision': float(mean_prec) if mean_prec is not None else None,
                        'std_precision': float(std_prec) if std_prec is not None else None,
                        'mean_recall': float(mean_rec) if mean_rec is not None else None,
                        'std_recall': float(std_rec) if std_rec is not None else None,
                        'mean_f1_score': float(mean_f1) if mean_f1 is not None else None,
                        'std_f1_score': float(std_f1) if std_f1 is not None else None,
                        'count': count,
                        'scores': scores
                    }
            
            # Calculate overall statistics
            if experiment_stats:
                all_means = [stats['mean'] for stats in experiment_stats.values()]
                overall_mean = np.mean(all_means)
                overall_std = np.std(all_means)
                
                results[model][experiment_type] = {
                    'overall_mean': float(overall_mean),
                    'overall_std': float(overall_std),
                    'experiment_stats': experiment_stats,
                    'total_parameter_values': len(experiment_stats)
                }
            else:
                results[model][experiment_type] = {
                    'overall_mean': None,
                    'overall_std': None,
                    'experiment_stats': {},
                    'total_parameter_values': 0
                }
    
    return results

def save_average_scores_per_n_examples(results):
    """Save average scores per n_examples for each model and approach."""
    output_dir = "results/mean_scores/n_examples/raw_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for model, experiment_types in results.items():
        for experiment_type in ["mean_activations_examples", "conceptor_examples"]:
            if experiment_type in experiment_types and experiment_types[experiment_type]["experiment_stats"]:
                stats = experiment_types[experiment_type]["experiment_stats"]
                
                # Create simplified data structure for n_examples
                n_examples_data = {}
                for n_examples, stat in stats.items():
                    n_examples_data[str(n_examples)] = {
                        'mean': stat['mean'],
                        'std': stat['std'],
                        'count': stat['count']
                    }
                
                # Save to JSON file
                output_file = os.path.join(output_dir, f"{model}_{experiment_type}_average_scores.json")
                with open(output_file, 'w') as f:
                    json.dump(n_examples_data, f, indent=2)
                print(f"Saved average scores per n_examples: {output_file}")

def save_average_scores_per_layers(results):
    """Save average scores per layers for each model and approach."""
    output_dir = "results/mean_scores/layers/raw_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for model, experiment_types in results.items():
        for experiment_type in ["mean_activations_layers", "conceptor_layers"]:
            if experiment_type in experiment_types and experiment_types[experiment_type]["experiment_stats"]:
                stats = experiment_types[experiment_type]["experiment_stats"]
                
                # Create simplified data structure for layers
                layers_data = {}
                for layer, stat in stats.items():
                    layers_data[str(layer)] = {
                        'mean': stat['mean'],
                        'std': stat['std'],
                        'count': stat['count']
                    }
                
                # Save to JSON file
                output_file = os.path.join(output_dir, f"{model}_{experiment_type}_average_scores.json")
                with open(output_file, 'w') as f:
                    json.dump(layers_data, f, indent=2)
                print(f"Saved average scores per layers: {output_file}")

def save_best_n_examples_per_model(results):
    """Save the n_examples value with the highest average accuracy per model and approach, including all metrics, in separate files per model."""
    output_dir = "results/mean_scores/n_examples/raw_results"
    os.makedirs(output_dir, exist_ok=True)
    best_n_examples = {}
    for model, experiment_types in results.items():
        best_n_examples[model] = {}
        for experiment_type in ["mean_activations_examples", "conceptor_examples"]:
            if experiment_type in experiment_types and experiment_types[experiment_type]["experiment_stats"]:
                stats = experiment_types[experiment_type]["experiment_stats"]
                # Find the n_examples with the highest mean accuracy
                best_n = None
                best_mean = -float('inf')
                for n_examples, stat in stats.items():
                    if stat['mean'] > best_mean:
                        best_mean = stat['mean']
                        best_n = n_examples
                if best_n is not None:
                    stat = stats[best_n]
                    best_n_examples[model][experiment_type] = {
                        "best_n_examples": best_n,
                        "mean_accuracy": stat.get("mean"),
                        "mean_precision": stat.get("mean_precision"),
                        "mean_recall": stat.get("mean_recall"),
                        "mean_f1_score": stat.get("mean_f1_score")
                    }
        # Save per-model JSON file
        model_file = os.path.join(output_dir, f"best_n_examples_{model}.json")
        with open(model_file, 'w') as f:
            json.dump({model: best_n_examples[model]}, f, indent=2)
        print(f"Saved best n_examples for model {model}: {model_file}")
    # Optionally, also save the combined file as before
    combined_file = os.path.join(output_dir, "best_n_examples_per_model.json")
    with open(combined_file, 'w') as f:
        json.dump(best_n_examples, f, indent=2)
    print(f"Saved combined best n_examples per model: {combined_file}")

def main():
    print("Calculating mean scores across all seeds for all experiments...")
    print("=" * 80)
    
    # Extract all data
    all_data = extract_experiment_data()
    
    # Calculate mean scores
    results = calculate_mean_scores(all_data)
    
    # Save average scores per n_examples
    save_average_scores_per_n_examples(results)
    
    # Save average scores per layers
    save_average_scores_per_layers(results)
    
    # Save best n_examples per model
    save_best_n_examples_per_model(results)
    
    print("\n" + "=" * 80)
    print("MEAN SCORES ACROSS SEEDS")
    print("=" * 80)
    
    for model, experiment_types in results.items():
        print(f"\n{model}:")
        print("-" * 50)
        
        for experiment_type, stats in experiment_types.items():
            if stats['total_parameter_values'] > 0:
                print(f"  {experiment_type}:")
                print(f"    Overall mean: {stats['overall_mean']:.4f} ± {stats['overall_std']:.4f}")
                print(f"    Parameter values: {stats['total_parameter_values']}")
                
                # Show a few examples
                param_list = sorted(stats['experiment_stats'].keys())
                print(f"    Examples:")
                for param in param_list[:3]:  # Show first 3
                    param_stats = stats['experiment_stats'][param]
                    print(f"      {param}: {param_stats['mean']:.4f} ± {param_stats['std']:.4f} (n={param_stats['count']})")
                if len(param_list) > 3:
                    print(f"      ... and {len(param_list) - 3} more values")
            else:
                print(f"  {experiment_type}: No data found")
    
    # Save overall results
    output_file = "results/mean_scores/raw_results/mean_scores_across_seeds.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nOverall results saved to {output_file}")

if __name__ == "__main__":
    main() 