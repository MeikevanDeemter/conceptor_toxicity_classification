import torch
import os
import json
import create_plots
from store_settings import update_optimal_settings, get_optimal_settings

def store_results(filename, scores, model_name, experiment_type, module_name, n_examples, aperture, seed):
    """
    Store the results in a structured JSON file.
    
    Args:
        filename (str): The filename to save the results to
        scores (dict): Dictionary containing validation scores
        model_name (str): Name of the model
        experiment_type (str): Type of experiment (examples, aperture, layers, confusion_matrix)
        module_name (str): Name of the module being tested
        n_examples (int): Number of examples used
        aperture (float): Aperture value used
        seed (int): Seed value for the experiment
    """
    
    # Sanitize model name for filename
    safe_model_name = model_name.replace('/', '-').replace('\\', '-')
    
    # Create a structured results dictionary
    structured_results = {
        "model": {
            "name": model_name,
            "dataset_seed": seed
        },
        "experiment": {
            "type": experiment_type,
            "module": module_name
        },
        "metrics": {
            "validation_scores": scores.get('validation_set_score', {}),
            "training_scores": scores.get('training_set_score', {}),
            "test_scores": scores.get('test_set_score', {})
        },
        "parameters": {}
    }
    
    # Add parameters based on experiment type
    if experiment_type == "mean_activations_layers":
        # For layers experiment, include the aperture used
        structured_results["parameters"]["n_examples"] = n_examples
        # The module names are already in the validation_scores keys
    elif experiment_type == "mean_activations_aperture":
        # For aperture experiment, include the number of examples used
        structured_results["parameters"]["n_examples"] = n_examples
        structured_results["parameters"]["module_name"] = module_name
    elif experiment_type == "mean_activations_examples":
        # For examples experiment, include the aperture used
        structured_results["parameters"]["module_name"] = module_name
    elif experiment_type == "mean_activations_confusion_matrix":
        # For confusion matrix, include both aperture and module
        structured_results["parameters"].update({
            "module_name": module_name,
            "model_name": model_name,
            "n_examples": n_examples
        })
        # Add confusion matrix specific metrics
        structured_results["metrics"].update({
            "TP": scores.get('TP', None),
            "FP": scores.get('FP', None),
            "TN": scores.get('TN', None),
            "FN": scores.get('FN', None),
            "accuracy": scores.get('accuracy', None),
            "precision": scores.get('precision', None),
            "recall": scores.get('recall', None),
            "f1_score": scores.get('f1_score', None)
        })
    
    # Save to JSON file with proper formatting
    with open(filename, "w") as f:
        json.dump(structured_results, f, indent=2)
    
    
    print(f"Results saved to {filename}")



def mean_activation_similarity_score(x, mean_vec):
    """
    Compute cosine similarity between x and mean activation vector.
    """
    numerator = torch.dot(x, mean_vec)
    denominator = torch.norm(x) * torch.norm(mean_vec)
    return numerator / denominator if denominator != 0 else torch.tensor(0.0, device=x.device)

def mean_activation_similarity_difference(x, mean_toxic, mean_nontoxic):
    """
    Compute difference in similarity with toxic vs non-toxic mean activations.
    """
    tox_score = mean_activation_similarity_score(x, mean_toxic)
    nontox_score = mean_activation_similarity_score(x, mean_nontoxic)
    return tox_score - nontox_score

def calculate_average_score(test_set_activations, test_dataset, mean_toxic, mean_nontoxic):
    # Initialize counters
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Process test set
    for idx, activation in enumerate(test_set_activations):

        # Calculate the cosine similarity difference between the activation and the mean activations for this example.
        score = mean_activation_similarity_difference(activation, mean_toxic, mean_nontoxic)

        # Get the toxicity label for this example
        toxicity_label = test_dataset[idx]["labels"]["toxicity"]
        
        # Determine if the example is correctly classified
        if score > 0 and toxicity_label == 1:
            true_positives += 1
        elif score < 0 and toxicity_label == 0:
            true_negatives += 1
        elif score > 0 and toxicity_label == 0:
            false_positives += 1
        elif score < 0 and toxicity_label == 1:
            false_negatives += 1
    
    return true_positives, false_positives, true_negatives, false_negatives


def layers(args, scores, train_toxic_activations, train_non_toxic_activations, validation_set_activations, validation_dataset, seed):
    module_names = list(train_toxic_activations.keys())
    best_module_name = ""
    highest_accuracy = 0
    for module_name in module_names:
        scores['validation_set_score'][module_name] = {} # Initialize scores for each module
    
    for module_name in module_names:
        print(f"\nProcessing {args.experiment_type} with module {module_name}")
        # Get the activations for this layer of all examples
        acts_toxic = train_toxic_activations[module_name]
        acts_non_toxic = train_non_toxic_activations[module_name]
            
        # Calculate mean activations for toxic and non-toxic examples
        mean_acts_toxic = torch.mean(acts_toxic, dim=0)
        mean_acts_non_toxic = torch.mean(acts_non_toxic, dim=0)
        
        TP, FP, TN, FN = calculate_average_score(validation_set_activations[module_name],
                                        validation_dataset,
                                        mean_acts_toxic, 
                                        mean_acts_non_toxic)
            
        # Calculate the average scores
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Store the accuracy in the scores dictionary
        scores['validation_set_score'][module_name] = accuracy

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
         
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_module_name = module_name
        elif accuracy == highest_accuracy:
            best_module_name = best_module_name + ", " + module_name # if there are multiple modules with the same accuracy

    # Create results directory for the model if it doesn't exist
    os.makedirs(f"results/seed_{seed}/{args.experiment_type}/{args.model_name}/raw_results", exist_ok=True)
    
    # Create plot after processing all layers
    create_plots.accuracy_vs_layers(scores, module_names, None, args.model_name, seed, args.experiment_type)
    filename = f"results/seed_{seed}/{args.experiment_type}/{args.model_name}/raw_results/layer_testing_{args.model_name}.json"
    store_results(filename, scores, args.model_name, args.experiment_type, module_name, args.n_examples, None, seed)
    print(f"Best module: {best_module_name} with accuracy: {highest_accuracy:.4f}")
    
    # Store optimal settings
    update_optimal_settings(
        model_name=args.model_name,
        seed=seed,
        layer=best_module_name,
        aperture=None,
        accuracy=highest_accuracy,
        experiment_type="mean_activations_layers",
        notes="Best performing layer for toxic classification using mean activations"
        )
    
    
def examples(args, scores, train_toxic_activations, train_non_toxic_activations, test_set_activations, test_dataset, seed):
    # Get the best layer from optimal settings
    settings = get_optimal_settings(args.model_name, seed, "mean_activation_based")
    module_name = settings['best_layer']
    if "," in module_name:
        module_name = module_name.split(", ")[-1].strip()  # Take the last layer if multiple
    
    args.n_examples = [5,10,20,30,40,50,100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000]
    for n_examples in args.n_examples:
        scores['test_set_score'][n_examples] = {} # Initialize scores for each n_examples
    
    for n_examples in args.n_examples:
        print(f"\nProcessing {args.experiment_type} for seed {seed} with {n_examples} examples")
        # Get the activations for this layer
        acts_toxic = train_toxic_activations[module_name][:n_examples]
        acts_non_toxic = train_non_toxic_activations[module_name][:n_examples]
        
        # Calculate mean activations for toxic and non-toxic examples
        mean_acts_toxic = torch.mean(acts_toxic, dim=0)
        mean_acts_non_toxic = torch.mean(acts_non_toxic, dim=0)
    
        TP, FP, TN, FN = calculate_average_score(test_set_activations[module_name],
                                        test_dataset,
                                        mean_acts_toxic, 
                                        mean_acts_non_toxic)
        
        # Calculate the average scores
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Store all metrics in the scores dictionary
        scores['test_set_score'][n_examples] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")

    # Create results directory for the model if it doesn't exist
    os.makedirs(f"results/seed_{seed}/{args.experiment_type}/{args.model_name}/raw_results", exist_ok=True)
    
    # Prepare a flat dict for plotting: n_examples -> accuracy
    flat_scores = {'test_set_score': {n: scores['test_set_score'][n]['accuracy'] for n in args.n_examples}}
    create_plots.accuracy_vs_examples(flat_scores, args.n_examples, module_name, None, args.model_name, seed, args.experiment_type)
    filename = f"results/seed_{seed}/{args.experiment_type}/{args.model_name}/raw_results/module_{module_name}.json"
    store_results(filename, scores, args.model_name, args.experiment_type, module_name, args.n_examples, None, seed)

def confusion_matrix(args, train_toxic_activations, train_non_toxic_activations, test_set_activations, test_dataset, seed):
    import json
    # Load best_n_examples from JSON
    with open("results/mean_scores/n_examples/best_n_examples_per_model.json", "r") as f:
        best_n_dict = json.load(f)
    model_name = args.model_name.replace("/", "-")
    best_n = None
    if model_name in best_n_dict and "mean_activations_examples" in best_n_dict[model_name]:
        best_n = best_n_dict[model_name]["mean_activations_examples"].get("best_n_examples", None)
    if best_n is None:
        raise ValueError(f"No best_n_examples found for {model_name} in mean activations confusion_matrix.")
    best_n = int(best_n)

    settings = get_optimal_settings(args.model_name, seed, "mean_activation_based")
    module_name = settings['best_layer']
    if "," in module_name:
        module_name = module_name.split(", ")[-1].strip()  # Take the last layer if multiple
    print(f"\nProcessing {args.experiment_type} for seed {seed} with module {module_name}")
    scores = {} # TP, FP, TN, FN, accuracy, precision, recall, f1_score
    # Get the activations for this layer of all examples
    n_examples = best_n
    acts_toxic = train_toxic_activations[module_name][:n_examples]
    acts_non_toxic = train_non_toxic_activations[module_name][:n_examples]
    
    # Calculate mean activations for toxic and non-toxic examples
    mean_acts_toxic = torch.mean(acts_toxic, dim=0)
    mean_acts_non_toxic = torch.mean(acts_non_toxic, dim=0)
    
    TP, FP, TN, FN = calculate_average_score(test_set_activations[module_name],
                                    test_dataset,
                                    mean_acts_toxic, 
                                    mean_acts_non_toxic)
        
    # Calculate the average scores
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Store the scores in the scores dictionary
    scores['TP'] = TP
    scores['FP'] = FP
    scores['TN'] = TN
    scores['FN'] = FN
    scores['accuracy'] = accuracy
    scores['precision'] = precision
    scores['recall'] = recall
    scores['f1_score'] = f1_score
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
    
    # Create results directory for the model if it doesn't exist
    os.makedirs(f"results/seed_{seed}/{args.experiment_type}_for_best_n_examples/{args.model_name}/raw_results", exist_ok=True)
    
    # Create plot after processing all layers
    create_plots.confusion_matrix(TP, FP, TN, FN, None, args.model_name, module_name, seed, args.experiment_type)
    filename = f"results/seed_{seed}/{args.experiment_type}_for_best_n_examples/{args.model_name}/raw_results/module_{module_name}.json"
    store_results(filename, scores, args.model_name, args.experiment_type, module_name, n_examples, None, seed)
    
