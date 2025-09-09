import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json        


def ensure_results_dir(model_name, seed, results_dir):
    """Ensure the results directory exists."""
    os.makedirs(results_dir, exist_ok=True)
    return results_dir
    

def accuracy_vs_examples(scores, n_examples, module_name, aperture, model_name, seed, experiment_type="conceptor_examples"):
    """Plot accuracy vs number of examples."""
    results_dir = f"results/seed_{seed}/{experiment_type}/{model_name}/plots"
    results_dir = ensure_results_dir(model_name, seed, results_dir)
    plt.figure(figsize=(10, 6))
    plt.plot(n_examples, [scores['test_set_score'][n] for n in n_examples], 'b-o', label='Test')
    plt.xlabel('Number of Examples', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title(f'Classification Accuracy vs Number of Examples\n{module_name}', fontsize=18)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"classification_accuracy_vs_examples_{model_name}_with_module_{module_name}_and_aperture_{aperture}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def accuracy_vs_aperture(scores, aperture_values, module_name, aperture, model_name, seed, aperture_experiment, experiment_type="conceptor_aperture"):
    """Plot accuracy vs aperture values."""
    results_dir = f"results/seed_{seed}/{experiment_type}/{model_name}/plots"
    results_dir = ensure_results_dir(model_name, seed, results_dir)
    plt.figure(figsize=(10, 6))
    
    # Handle the new nested scores structure
    if isinstance(scores['validation_set_score'], dict) and len(scores['validation_set_score']) > 0:
        # Check if the first value is also a dict (nested structure)
        first_key = list(scores['validation_set_score'].keys())[0]
        if isinstance(scores['validation_set_score'][first_key], dict):
            # This is the new nested structure: scores[toxic_aperture][non_toxic_aperture]
            # For plotting, we'll use the diagonal values (same toxic and non-toxic aperture)
            diagonal_scores = []
            diagonal_apertures = []
            
            for a in aperture_values:
                if a in scores['validation_set_score'] and a in scores['validation_set_score'][a]:
                    diagonal_scores.append(scores['validation_set_score'][a][a])
                    diagonal_apertures.append(a)
            
            if diagonal_scores:
                plt.semilogx(diagonal_apertures, diagonal_scores, 'b-o', label='Validation (Diagonal)')
        else:
            # This is the old flat structure
            plt.semilogx(aperture_values, [scores['validation_set_score'][a] for a in aperture_values], 'b-o', label='Validation')
    else:
        print("Warning: No validation scores found for plotting")
        return
    
    plt.xlabel('Aperture Value', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title(f'Classification Accuracy vs Aperture Value\n{module_name}', fontsize=18)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, f"classification_accuracy_vs_aperture_{model_name}_with_module_{module_name}_with_{aperture_experiment}.png"), dpi=300, bbox_inches='tight')
    plt.close()

def accuracy_vs_layers_line(scores, module_names, aperture, model_name, seed, experiment_type="conceptor_layers"):
    """Plot accuracy vs layers as a line graph with separate lines for each module type."""
    results_dir = f"results/seed_{seed}/{experiment_type}/{model_name}/plots"
    results_dir = ensure_results_dir(model_name, seed, results_dir)
    plt.figure(figsize=(15, 8))
    
    # Extract module types and layer numbers
    module_types = set()
    layer_data = {}
    
    for name in module_names:
        parts = name.split('.')
        layer_num = int(parts[2])
        module_type = parts[3]  # e.g., 'ln_1', 'ln_2'
        module_types.add(module_type)
        
        if module_type not in layer_data:
            layer_data[module_type] = {'layers': [], 'scores': []}
        
        layer_data[module_type]['layers'].append(layer_num)
        layer_data[module_type]['scores'].append(scores['validation_set_score'][name])
    
    # Sort data for each module type
    for module_type in layer_data:
        sorted_indices = np.argsort(layer_data[module_type]['layers'])
        layer_data[module_type]['layers'] = [layer_data[module_type]['layers'][i] for i in sorted_indices]
        layer_data[module_type]['scores'] = [layer_data[module_type]['scores'][i] for i in sorted_indices]
    
    # Create a line for each module type with different colors
    colors = plt.cm.Set2(np.linspace(0, 1, len(module_types)))  # Use Set2 colormap for distinct colors
    for (module_type, data), color in zip(layer_data.items(), colors):
        plt.plot(data['layers'], data['scores'], 'o-', label=module_type, color=color, linewidth=2, markersize=6)
    
    # Set x-axis to show all layer numbers
    all_layers = sorted(set([layer for data in layer_data.values() for layer in data['layers']]))
    plt.xlabel('Layer Number', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title('Classification Accuracy vs Layer by Module Type', fontsize=18)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
    plt.xticks(all_layers, all_layers, fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"classification_accuracy_vs_layers_line_{model_name}_with_aperture_{aperture}_zoom_in.png"), dpi=300, bbox_inches='tight')
    plt.close()

def accuracy_vs_layers(scores, module_names, aperture, model_name, seed, experiment_type="conceptor_layers"):
    """Plot accuracy vs layers as both bar and line graphs."""
    results_dir = f"results/seed_{seed}/{experiment_type}/{model_name}/plots"
    results_dir = ensure_results_dir(model_name, seed, results_dir)
    plt.figure(figsize=(12, 6))
    validation_scores = [scores['validation_set_score'][m] for m in module_names]
    plt.bar(range(len(module_names)), validation_scores, label='Validation')
    plt.xlabel('Layer', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title('Classification Accuracy vs Layer', fontsize=18)
    plt.xticks(range(len(module_names)), module_names, rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=13)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"classification_accuracy_vs_layers_{model_name}_with_aperture.png"), dpi=300)
    plt.close()
    
    # Create line plot
    accuracy_vs_layers_line(scores, module_names, aperture, model_name, seed, experiment_type)

def confusion_matrix(tp, fp, tn, fn, aperture, model_name, module_name, seed, experiment_type="conceptor_confusion_matrix"):
    """Plot confusion matrix."""
    results_dir = f"results/seed_{seed}/{experiment_type}_for_best_n_examples/{model_name}/plots"
    results_dir = ensure_results_dir(model_name, seed, results_dir)
    plt.figure(figsize=(8, 6))
    cm = np.array([[tp, fp], [fn, tn]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Toxic', 'Predicted Non-toxic'],
                yticklabels=['Actual Toxic', 'Actual Non-toxic'],
                annot_kws={"size": 15}, cbar_kws={'label': 'Count'})
    
    # Create a more descriptive title that includes the experiment type
    experiment_display_name = experiment_type.replace('_', ' ').title()
    title = f'Confusion Matrix - {experiment_display_name}\n{model_name} - {module_name}'
    if aperture is not None:
        title += f' (Aperture: {aperture})'
    title += f'\nSeed: {seed}'
    
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(os.path.join(results_dir, f"confusion_matrix_for_{module_name}_with_aperture_{aperture}_{model_name}.png"), dpi=300)
    plt.close()

def plotting_stored_results(seed):
    """Plot examples vs accuracy for all three GPT-Neo models on the same plot."""

    files = {
        "conceptor_examples": "/home/meike/conceptor-llm/bachelor-thesis-meike/results/seed_42/conceptor_examples/EleutherAI-gpt-neo-2.7B/raw_results/module_transformer.h.31.ln_2_with_aperture_[0.0145].json",
        "mean_activations_examples": "/home/meike/conceptor-llm/bachelor-thesis-meike/results/seed_42/mean_activations_examples/EleutherAI-gpt-neo-2.7B/raw_results/mean_activations_examples_module_transformer.h.28.mlp_EleutherAI-gpt-neo-2.7B.json"
    }
    plt.figure(figsize=(10, 6))
    colors = ['b', 'g', 'r']

    # Extract model and module info from the first file
    with open(list(files.values())[0]) as f:
        data = json.load(f)
        model_name = data['model']['name']
        module_name = data['experiment']['module']
        dataset_seed = data['model']['dataset_seed']

    for (label, file), color in zip(files.items(), colors):
        with open(file) as f:
            data = json.load(f)
        scores = data['metrics']['test_scores']
        x = [int(k) for k in scores.keys()]
        y = [scores[k] for k in scores.keys()]
        
        # Calculate standard deviation using a rolling window
        window_size = 3  # Use 3 points for standard deviation calculation
        std_dev = []
        for i in range(len(y)):
            start_idx = max(0, i - window_size + 1)
            window = y[start_idx:i+1]
            std_dev.append(np.std(window))
        
        # Plot mean with standard deviation
        plt.plot(x, y, marker='o', label=label, color=color)
        plt.fill_between(x, 
                        [y[i] - std_dev[i] for i in range(len(y))],
                        [y[i] + std_dev[i] for i in range(len(y))],
                        color=color, alpha=0.2)

    plt.xlabel('Number of Examples', fontsize=15)
    plt.ylabel('Test Accuracy', fontsize=15)
    plt.title(f'Accuracy vs Number of Examples: Conceptor vs Mean Activations\nModel: {model_name}, Module: {module_name}, Dataset Seed: {dataset_seed}', fontsize=18)
    plt.legend(fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Create results directory if it doesn't exist
    save_dir = f"/home/meike/conceptor-llm/bachelor-thesis-meike/results/seed_{seed}/conceptor_vs_mean_acts"
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(os.path.join(save_dir, f"examples_vs_accuracy_mean_activations_and_conceptor_{model_name}_{module_name}_seed_{dataset_seed}.png"), dpi=300)
    plt.close()
    
def aperture_heatmap(scores, aperture_values, module_name, model_name, seed, experiment_type="conceptor_aperture"):
    """Create a heatmap showing accuracy for all combinations of toxic and non-toxic apertures."""
    results_dir = f"results/seed_{seed}/{experiment_type}/{model_name}/plots"
    results_dir = ensure_results_dir(model_name, seed, results_dir)
    
    # Create a 2D array for the heatmap
    heatmap_data = np.zeros((len(aperture_values), len(aperture_values)))
    
    # Fill the heatmap with accuracy values
    for i, toxic_aperture in enumerate(aperture_values):
        for j, non_toxic_aperture in enumerate(aperture_values):
            if (toxic_aperture in scores['validation_set_score'] and 
                non_toxic_aperture in scores['validation_set_score'][toxic_aperture]):
                heatmap_data[i, j] = scores['validation_set_score'][toxic_aperture][non_toxic_aperture]
            else:
                heatmap_data[i, j] = np.nan
    
    plt.figure(figsize=(12, 10))
    
    # Create the heatmap
    sns.heatmap(heatmap_data, 
                xticklabels=[f'{a:.2e}' for a in aperture_values],
                yticklabels=[f'{a:.2e}' for a in aperture_values],
                annot=True, 
                fmt='.3f',
                cmap='viridis',
                cbar_kws={'label': 'Accuracy'},
                annot_kws={"size": 13})
    
    plt.xlabel('Non-toxic Aperture', fontsize=15)
    plt.ylabel('Toxic Aperture', fontsize=15)
    plt.title(f'Aperture Combination Heatmap\n{module_name}', fontsize=18)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"aperture_heatmap_{model_name}_with_module_{module_name}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    