import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, '..', 'mean_scores_across_seeds.json')
plots_dir = base_dir
os.makedirs(plots_dir, exist_ok=True)

# Model info: (model_name, color, label)
model_info = [
    ("EleutherAI-gpt-neo-125M", 'blue', 'GPT-Neo 125M'),
    ("EleutherAI-gpt-neo-1.3B", 'green', 'GPT-Neo 1.3B'),
    ("EleutherAI-gpt-neo-2.7B", 'red', 'GPT-Neo 2.7B'),
]

# Load data
with open(json_path, 'r') as f:
    results = json.load(f)

# Plot for mean activations
plt.figure(figsize=(10, 6))
plotted_any_ma = False
for model, color, label in model_info:
    experiment_types = results.get(model, {})
    if 'mean_activations_examples' in experiment_types and experiment_types['mean_activations_examples']['experiment_stats']:
        stats = experiment_types['mean_activations_examples']['experiment_stats']
        n_examples = sorted([int(k) for k in stats.keys() if 20 <= int(k) <= 3000])
        if n_examples:
            means = [stats[str(n)]['mean'] * 100 for n in n_examples]
            stds = [stats[str(n)]['std'] * 100 for n in n_examples]
            plt.plot(n_examples, means, color=color, label=label)
            plt.scatter(n_examples, means, color=color, marker='o')
            plt.fill_between(n_examples, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=color, alpha=0.2)
            plotted_any_ma = True
if plotted_any_ma:
    plt.xlabel('Number of Examples')
    plt.ylabel('Average Accuracy (%) ± Std Dev')
    plt.title('Model Comparison: Average Accuracy vs Number of Examples\nMean Activations Approach')
    plt.grid(True)
    plt.legend()
    # Make outer borders thinner
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.25)
        spine.set_edgecolor('#888888')
    plt.tight_layout()
    plot_path_png = os.path.join(plots_dir, 'model_comparison_mean_activations.png')
    plot_path_pdf = os.path.join(plots_dir, 'model_comparison_mean_activations.pdf')
    plt.savefig(plot_path_png, dpi=300)
    plt.savefig(plot_path_pdf)
    plt.close()
    print(f'Saved plot: {plot_path_png}')
    print(f'Saved plot: {plot_path_pdf}')
else:
    print('No mean activations data available for plotting in the specified n_examples range.')

# Plot for conceptor based
plt.figure(figsize=(10, 6))
plotted_any = False
for model, color, label in model_info:
    experiment_types = results.get(model, {})
    if 'conceptor_examples' in experiment_types and experiment_types['conceptor_examples']['experiment_stats']:
        stats = experiment_types['conceptor_examples']['experiment_stats']
        n_examples = sorted([int(k) for k in stats.keys() if 20 <= int(k) <= 3000])
        if n_examples:
            means = [stats[str(n)]['mean'] * 100 for n in n_examples]
            stds = [stats[str(n)]['std'] * 100 for n in n_examples]
            plt.plot(n_examples, means, color=color, label=label)
            plt.scatter(n_examples, means, color=color, marker='o')
            plt.fill_between(n_examples, np.array(means) - np.array(stds), np.array(means) + np.array(stds), color=color, alpha=0.2)
            plotted_any = True
if plotted_any:
    plt.xlabel('Number of Examples')
    plt.ylabel('Average Accuracy (%) ± Std Dev')
    plt.title('Model Comparison: Average Accuracy vs Number of Examples\nConceptor Based Approach')
    plt.grid(True)
    plt.legend()
    # Make outer borders thinner
    for spine in plt.gca().spines.values():
        spine.set_linewidth(0.25)
        spine.set_edgecolor('#888888')
    plt.tight_layout()
    plot_path_png = os.path.join(plots_dir, 'model_comparison_conceptor.png')
    plot_path_pdf = os.path.join(plots_dir, 'model_comparison_conceptor.pdf')
    plt.savefig(plot_path_png, dpi=300)
    plt.savefig(plot_path_pdf)
    plt.close()
    print(f'Saved plot: {plot_path_png}')
    print(f'Saved plot: {plot_path_pdf}')
else:
    print('No conceptor-based data available for plotting in the specified n_examples range.') 