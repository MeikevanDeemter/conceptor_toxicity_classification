import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, '..', 'mean_scores_across_seeds.json')
plots_dir = base_dir
os.makedirs(plots_dir, exist_ok=True)

# Model info: (model_name, color)
model_info = [
    ("EleutherAI-gpt-neo-125M", 'blue'),
    ("EleutherAI-gpt-neo-1.3B", 'green'),
    ("EleutherAI-gpt-neo-2.7B", 'red'),
]

# Load data
with open(json_path, 'r') as f:
    results = json.load(f)

for model, color in model_info:
    experiment_types = results.get(model, {})
    plt.figure(figsize=(10, 6))
    plotted_any = False
    # Plot mean activations layers
    if 'mean_activations_layers' in experiment_types and experiment_types['mean_activations_layers']['experiment_stats']:
        stats = experiment_types['mean_activations_layers']['experiment_stats']
        layers = sorted([int(k) for k in stats.keys()])
        if layers:
            means = [stats[str(l)]['mean'] * 100 for l in layers]
            stds = [stats[str(l)]['std'] * 100 for l in layers]
            plt.plot(layers, means, color=color, label='Mean Activations')
            plt.scatter(layers, means, color=color, marker='o')
            plt.fill_between(layers, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=color, alpha=0.2)
            plotted_any = True
    # Plot conceptor layers
    if 'conceptor_layers' in experiment_types and experiment_types['conceptor_layers']['experiment_stats']:
        stats = experiment_types['conceptor_layers']['experiment_stats']
        layers = sorted([int(k) for k in stats.keys()])
        if layers:
            means = [stats[str(l)]['mean'] * 100 for l in layers]
            stds = [stats[str(l)]['std'] * 100 for l in layers]
            plt.plot(layers, means, color='orange', label='Conceptor Based')
            plt.scatter(layers, means, color='orange', marker='o')
            plt.fill_between(layers, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color='orange', alpha=0.2)
            plotted_any = True
    if plotted_any:
        plt.xlabel('Layer')
        plt.ylabel('Average Accuracy (%) ± Std Dev')
        plt.title(f'Layer Comparison: Average Accuracy per Layer\n{model}')
        plt.grid(True)
        plt.legend()
        # Make outer borders thin and grey
        for spine in plt.gca().spines.values():
            spine.set_linewidth(0.25)
            spine.set_edgecolor('#888888')
        plt.tight_layout()
        model_size = model.split('-')[-1]
        plot_path_png = os.path.join(plots_dir, f'{model_size}_layer_comparison.png')
        plot_path_pdf = os.path.join(plots_dir, f'{model_size}_layer_comparison.pdf')
        plt.savefig(plot_path_png, dpi=300)
        plt.savefig(plot_path_pdf)
        plt.close()
        print(f'Saved plot: {plot_path_png}')
        print(f'Saved plot: {plot_path_pdf}')
    else:
        print(f'No layer data available for {model}.') 