import os
import json
import matplotlib.pyplot as plt
import numpy as np
import re

# Directories
n_examples_dir = 'results/mean_scores/n_examples/raw_results'
layers_dir = 'results/mean_scores/layers/raw_results'
n_examples_plots_dir = 'results/mean_scores/n_examples/plots'
layers_plots_dir = 'results/mean_scores/layers/plots'
os.makedirs(n_examples_plots_dir, exist_ok=True)
os.makedirs(layers_plots_dir, exist_ok=True)

# Helper to extract model names from files
model_pattern = re.compile(r'EleutherAI-gpt-neo-[^_]+')

def get_models():
    models = set()
    for fname in os.listdir(n_examples_dir):
        match = model_pattern.search(fname)
        if match:
            models.add(match.group(0))
    return sorted(models)

def plot_n_examples(model):
    approaches = ['conceptor_examples', 'mean_activations_examples']
    colors = {'conceptor_examples': 'blue', 'mean_activations_examples': 'green'}
    labels = {'conceptor_examples': 'Conceptor Based', 'mean_activations_examples': 'Mean Activation Approach'}
    plt.figure(figsize=(10, 6))
    plotted = False
    best_conceptor = None
    best_mean_activations = None
    for approach in approaches:
        fname = f'{model}_{approach}_average_scores.json'
        fpath = os.path.join(n_examples_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r') as f:
            data = json.load(f)
        n_examples = sorted([int(k) for k in data.keys()])
        filtered = [(n, data[str(n)]['mean'] * 100, data[str(n)]['std'] * 100)
                    for n in n_examples if 20 <= n <= 3000]
        if not filtered:
            continue
        x, y, yerr = zip(*filtered)
        plt.plot(x, y, color=colors[approach], label=labels[approach])
        plt.scatter(x, y, color=colors[approach], marker='o')
        plt.fill_between(x, np.array(y) - np.array(yerr), np.array(y) + np.array(yerr), color=colors[approach], alpha=0.2)
        if approach == 'conceptor_examples':
            best_conceptor = max(y)
        if approach == 'mean_activations_examples':
            best_mean_activations = max(y)
        plotted = True
    if plotted:
        if best_conceptor is not None:
            plt.axhline(best_conceptor, color='blue', linestyle='--', linewidth=1.0, label='Best Conceptor Score')
        if best_mean_activations is not None:
            plt.axhline(best_mean_activations, color='green', linestyle='--', linewidth=1.0, label='Best Mean Activations Score')
        plt.xlabel('Number of Examples', fontsize=15)
        plt.ylabel('Average Accuracy (%) ± Std Dev', fontsize=15)
        plt.title('Conceptor vs Mean Activation: Classification Accuracy Comparison' + f'\n{model}', fontsize=18)
        plt.grid(True)
        plt.legend(fontsize=13)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(0.25)
            spine.set_edgecolor('#888888')
        plt.tight_layout()
        model_size = model.split('-')[-1]
        plot_path_base = os.path.join(n_examples_plots_dir, f'{model_size}_conceptor_vs_mean_acts')
        plt.savefig(plot_path_base + '.png', dpi=300)
        plt.savefig(plot_path_base + '.pdf')
        plt.close()
        print(f'Saved plot: {plot_path_base}.png')
        print(f'Saved plot: {plot_path_base}.pdf')

def plot_layers_separate(model):
    approaches = ['conceptor_layers', 'mean_activations_layers']
    colors = {'conceptor_layers': 'blue', 'mean_activations_layers': 'green'}
    labels = {'conceptor_layers': 'Conceptor', 'mean_activations_layers': 'Mean Activation'}
    module_types = ['ln_1', 'attn', 'ln_2', 'mlp']
    # Use visually distinct, colorblind-friendly colors
    module_colors = {'ln_1': 'orange', 'attn': 'blue', 'ln_2': 'green', 'mlp': 'purple'}
    for approach in approaches:
        fname = f'{model}_{approach}_average_scores.json'
        fpath = os.path.join(layers_dir, fname)
        if not os.path.exists(fpath):
            continue
        with open(fpath, 'r') as f:
            data = json.load(f)
        def layer_num(layer_name):
            m = re.search(r'h\.(\d+)', layer_name)
            return int(m.group(1)) if m else -1
        # For each module, collect (layer_num, mean, std) tuples
        module_points = {m: [] for m in module_types}
        for layer_name, v in data.items():
            for m in module_types:
                if m in layer_name:
                    module_points[m].append((layer_num(layer_name), v['mean'] * 100, v['std'] * 100))
        plt.figure(figsize=(12, 6))
        all_layer_nums = set()
        for m in module_types:
            pts = sorted(module_points[m], key=lambda x: x[0])
            if pts:
                x = [p[0] for p in pts]
                y = [p[1] for p in pts]
                yerr = [p[2] for p in pts]
                all_layer_nums.update(x)
                plt.plot(x, y, color=module_colors[m], label=f'{m}')
                plt.scatter(x, y, color=module_colors[m], marker='o')
                plt.fill_between(x, np.array(y) - np.array(yerr), np.array(y) + np.array(yerr), color=module_colors[m], alpha=0.2)
        plt.xlabel('Layer Number', fontsize=17)
        plt.ylabel('Average Accuracy (%) ± Std Dev', fontsize=17)
        plt.title('Layerwise Classification Accuracy per Module' + f'\n{model} - {labels[approach]}', fontsize=20)
        plt.xticks(sorted(all_layer_nums), fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=15)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(0.25)
            spine.set_edgecolor('#888888')
        plt.tight_layout()
        plot_path_base = os.path.join(layers_plots_dir, f'{model}_{approach}_layer_vs_accuracy')
        plt.savefig(plot_path_base + '.png', dpi=300)
        plt.savefig(plot_path_base + '.pdf')
        plt.close()
        print(f'Saved plot: {plot_path_base}.png')
        print(f'Saved plot: {plot_path_base}.pdf')

def plot_model_comparison():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, 'results/mean_scores/raw_results/mean_scores_across_seeds.json')
    plots_dir = os.path.join(base_dir, 'results/mean_scores/compare_models')
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
                # Add horizontal line for best score
                best_score = max(means)
                plt.axhline(best_score, color=color, linestyle='--', linewidth=1.0, label=f'Best {label} Score')
                plotted_any_ma = True
    if plotted_any_ma:
        plt.xlabel('Number of Examples', fontsize=15)
        plt.ylabel('Average Accuracy (%) ± Std Dev', fontsize=15)
        plt.title('Model Comparison: Average Accuracy vs Number of Examples\nMean Activation Approach', fontsize=18)
        plt.grid(True)
        plt.legend(fontsize=13)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
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
                # Add horizontal line for best score
                best_score = max(means)
                plt.axhline(best_score, color=color, linestyle='--', linewidth=1.0, label=f'Best {label} Score')
                plotted_any = True
    if plotted_any:
        plt.xlabel('Number of Examples', fontsize=15)
        plt.ylabel('Average Accuracy (%) ± Std Dev', fontsize=15)
        plt.title('Model Comparison: Average Accuracy vs Number of Examples\nConceptor Approach', fontsize=18)
        plt.grid(True)
        plt.legend(fontsize=13)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
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

def main():
    models = get_models()
    for model in models:
        plot_n_examples(model)
        plot_layers_separate(model)
    plot_model_comparison()

if __name__ == '__main__':
    main() 