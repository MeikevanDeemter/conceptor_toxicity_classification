import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(base_dir, '..', 'mean_scores_across_seeds.json')
plots_dir = os.path.join(base_dir, 'plots')
raw_results_dir = os.path.join(base_dir, 'raw_results')
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(raw_results_dir, exist_ok=True)

# Load data
with open(json_path, 'r') as f:
    results = json.load(f)

# For each model, plot combined mean activations and conceptor examples
for model, experiment_types in results.items():
    data = {}
    for approach in ['mean_activations_examples', 'conceptor_examples']:
        if approach in experiment_types and experiment_types[approach]['experiment_stats']:
            stats = experiment_types[approach]['experiment_stats']
            n_examples = sorted([int(k) for k in stats.keys()])
            # Only keep n_examples in [20, 3000]
            filtered = [(n, stats[str(n)]['mean'] * 100, stats[str(n)]['std'] * 100)
                        for n in n_examples if 20 <= n <= 3000]
            if filtered:
                n_examples_filt, means_filt, stds_filt = zip(*filtered)
                data[approach] = {'n_examples': list(n_examples_filt), 'means': list(means_filt), 'stds': list(stds_filt)}

    # Only plot if both approaches have data
    if 'mean_activations_examples' in data and 'conceptor_examples' in data:
        plt.figure(figsize=(10, 6))
        # Mean Activations: green, dots for all points
        x_ma = data['mean_activations_examples']['n_examples']
        y_ma = data['mean_activations_examples']['means']
        std_ma = data['mean_activations_examples']['stds']
        plt.plot(x_ma, y_ma, color='green', label='Mean Activations')
        plt.scatter(x_ma, y_ma, color='green', marker='o')
        plt.fill_between(x_ma, np.array(y_ma) - np.array(std_ma), np.array(y_ma) + np.array(std_ma), color='green', alpha=0.2)
        # Conceptor: blue, dots for all points
        x_c = data['conceptor_examples']['n_examples']
        y_c = data['conceptor_examples']['means']
        std_c = data['conceptor_examples']['stds']
        plt.plot(x_c, y_c, color='blue', label='Conceptor Based')
        plt.scatter(x_c, y_c, color='blue', marker='o')
        plt.fill_between(x_c, np.array(y_c) - np.array(std_c), np.array(y_c) + np.array(std_c), color='blue', alpha=0.2)
        # Horizontal line for best conceptor score
        best_conceptor = max(y_c)
        plt.axhline(best_conceptor, color='blue', linestyle='--', linewidth=1.0, label='Best Conceptor Score')
        # Labels and title
        plt.xlabel('Number of Examples')
        plt.ylabel('Average Accuracy (%) ± Std Dev')
        plt.title('Conceptor vs Mean Activation: Classification Accuracy Comparison' + f'\n{model}')
        plt.grid(True)
        plt.legend()
        # Make outer borders thin and grey
        for spine in plt.gca().spines.values():
            spine.set_linewidth(0.25)
            spine.set_edgecolor('#888888')
        plt.tight_layout()
        # Extract model size for filename
        model_size = model.split('-')[-1]
        plot_path_png = os.path.join(plots_dir, f'{model_size}_conceptor_vs_mean_acts.png')
        plot_path_pdf = os.path.join(plots_dir, f'{model_size}_conceptor_vs_mean_acts.pdf')
        plt.savefig(plot_path_png, dpi=300)
        plt.savefig(plot_path_pdf)
        plt.close()
        print(f'Saved plot: {plot_path_png}')
        print(f'Saved plot: {plot_path_pdf}')

        # Optionally save the raw data used for plotting
        raw_data_path = os.path.join(raw_results_dir, f'{model}_combined_mean_vs_conceptor.json')
        with open(raw_data_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f'Saved raw data: {raw_data_path}') 