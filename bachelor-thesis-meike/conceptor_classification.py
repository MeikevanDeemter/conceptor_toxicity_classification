import create_plots
from hookedllm.steering_functions import compute_conceptor
import torch
import json
import os
from store_settings import update_optimal_settings, get_optimal_settings

    
def store_results(filename, scores, model_name, experiment_type, module_name, n_examples, aperture, seed):
    """
    Store the results in a structured JSON file.
    
    Args:
        scores (dict): Dictionary containing validation scores
        model_name (str): Name of the model
        experiment_type (str): Type of experiment (examples, aperture, layers, confusion_matrix)
        module_name (str): Name of the module being tested
        n_examples (int): Number of examples used
        aperture (float): Aperture value used
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
    if experiment_type == "conceptor_layers":
        # For layers experiment, include the aperture used
        structured_results["parameters"]["aperture"] = aperture
        structured_results["parameters"]["n_examples"] = n_examples
        # The module names are already in the validation_scores keys
    elif experiment_type == "conceptor_aperture":
        # For aperture experiment, include the number of examples used
        structured_results["parameters"]["n_examples"] = n_examples
        structured_results["parameters"]["module_name"] = module_name
    elif experiment_type == "conceptor_examples":
        # For examples experiment, include the aperture used
        structured_results["parameters"]["aperture"] = aperture
    elif experiment_type == "conceptor_confusion_matrix":
        # For confusion matrix, include both aperture and module
        structured_results["parameters"].update({
            "aperture": aperture,
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


def conceptor_similarity_score(x, C):
    """
    Compute cosine similarity between x and Cx, measuring how similar x is to the conceptor C.
    """
    Cx = C @ x # matrix multiplication
    numerator = torch.abs(torch.dot(x, Cx)) # dot product
    denominator = torch.norm(x) * torch.norm(Cx) # norm of x and Cx
    return numerator / denominator if denominator != 0 else torch.tensor(0.0, device=x.device) # avoid division by zero

def cosine_similarity_difference(x, C_toxic, C_nontoxic):
    """
    Compute difference in similarity with toxic vs non-toxic conceptors.
    """
    tox_score = conceptor_similarity_score(x, C_toxic) 
    nontox_score = conceptor_similarity_score(x, C_nontoxic)
    return tox_score - nontox_score


def calculate_average_score(test_set_activations, test_dataset, C_toxic, C_nontoxic):
    # Initialize counters
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    # Process test set
    for idx, activation in enumerate(test_set_activations):

        # Calculate the cosine similarity difference between the activation and the conceptors for this example.
        score = cosine_similarity_difference(activation, C_toxic, C_nontoxic)

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

def examples(args, scores, train_toxic_activations, train_non_toxic_activations, test_set_activations, test_dataset, seed):
    # set the number of examples to test (can be removed later, just for experiments)
    args.n_examples = [5,10,20,30,40,50,100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950, 2000, 2050, 2100, 2150, 2200, 2250, 2300, 2350, 2400, 2450, 2500, 2550, 2600, 2650, 2700, 2750, 2800, 2850, 2900, 2950, 3000]
    settings = get_optimal_settings(args.model_name, seed, "conceptor_based")
    module_name = settings['best_layer']
    if "," in module_name:
        module_name = module_name.split(", ")[-1].strip()  # Take the last layer if multiple
    aperture = settings['best_aperture']
    # Take the first aperture if multiple
    if isinstance(aperture, list):
        aperture = aperture[0]
    print(f"Using aperture: {aperture}")
    
    for n_examples in args.n_examples: # Initialize scores for each n_examples
        scores['test_set_score'][n_examples] = {} 
 
    for n_examples in args.n_examples:
        print(f"\nProcessing {args.experiment_type} for seed {seed} with {n_examples} examples")
        # Get the activations for this layer
        acts_toxic = train_toxic_activations[module_name][:n_examples]
        acts_non_toxic = train_non_toxic_activations[module_name][:n_examples]
        
        # Calculate toxic and non-toxic conceptors with the same aperture
        C_toxic = compute_conceptor(acts_toxic, aperture)
        C_nontoxic = compute_conceptor(acts_non_toxic, aperture)
    
        TP, FP, TN, FN = calculate_average_score(test_set_activations[module_name],
                                        test_dataset,
                                        C_toxic, 
                                        C_nontoxic)
        
        # Calculate the average scores
        print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

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
    filename = f"results/seed_{seed}/{args.experiment_type}/{args.model_name}/raw_results/module_{module_name}_aperture_{aperture}.json"
    # Prepare a flat dict for plotting: n_examples -> accuracy
    flat_scores = {'test_set_score': {n: scores['test_set_score'][n]['accuracy'] for n in args.n_examples}}
    create_plots.accuracy_vs_examples(flat_scores, args.n_examples, module_name, aperture, args.model_name, seed)
    store_results(filename, scores, args.model_name, args.experiment_type, module_name, args.n_examples, aperture, seed)
    
def aperture_zoom_in(args, scores, best_aperture, train_toxic_activations, train_non_toxic_activations, validation_set_activations, validation_dataset, seed):
    """ 
    Perform aperture zoom-in around the best aperture value (single aperture only).
    """
    print(f"\nStarting aperture zoom-in experiment for seed {seed}")
    settings = get_optimal_settings(args.model_name, seed, "conceptor_based")
    module_name = settings['best_layer']
    if "," in module_name:
        module_name = module_name.split(", ")[-1].strip()
    print(f"Best aperture found: {best_aperture}")

    # Create zoom-in scores dictionary (flat, not nested)
    zoom_scores = {'validation_set_score': {}}

    # Generate aperture ranges to check (±5 steps around best)
    apertures_to_check = []
    for current_best_aperture in best_aperture:
        for i in range(-5, 6):
            step_size = current_best_aperture / 10
            aperture = current_best_aperture + (i * step_size)
            if aperture > 0:
                apertures_to_check.append(round(aperture, 6))

    print(f"Apertures to check: {apertures_to_check}")
    highest_accuracy = 0
    best_aperture_final = 0

    # Test all apertures
    for aperture in apertures_to_check:
        print(f"\nTesting zoom-in aperture: Aperture={aperture}")
        acts_toxic = train_toxic_activations[module_name]
        acts_non_toxic = train_non_toxic_activations[module_name]
        C_toxic = compute_conceptor(acts_toxic, aperture)
        C_nontoxic = compute_conceptor(acts_non_toxic, aperture)
        TP, FP, TN, FN = calculate_average_score(validation_set_activations[module_name],
                                                validation_dataset,
                                                C_toxic, 
                                                C_nontoxic)
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        zoom_scores['validation_set_score'][aperture] = accuracy
        print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_aperture_final = aperture

    print(f"\nBest combination found:")
    print(f"  Aperture: {best_aperture_final}")
    print(f"  Accuracy: {highest_accuracy:.4f}")

    os.makedirs(f"results/seed_{seed}/{args.experiment_type}/{args.model_name}/raw_results", exist_ok=True)
    filename = f"results/seed_{seed}/{args.experiment_type}/{args.model_name}/raw_results/module_{module_name}_aperture_zoom_in.json"
    store_results(filename, zoom_scores, args.model_name, args.experiment_type, module_name, args.n_examples, best_aperture_final, seed)
    create_plots.accuracy_vs_aperture(zoom_scores, apertures_to_check, module_name, None, args.model_name, seed, "aperture_zoom_in", args.experiment_type)

    update_optimal_settings(
        model_name=args.model_name,
        seed=seed,
        layer=None,
        aperture=best_aperture_final,
        accuracy=highest_accuracy,
        experiment_type="conceptor_aperture_zoom_in",
        notes=f"Best performing aperture: {best_aperture_final}"
    )
    print(f"Zoom-in results saved to: {filename}")
    return best_aperture_final, highest_accuracy

def aperture(args, scores, train_toxic_activations, train_non_toxic_activations, validation_set_activations, validation_dataset, seed):
    """
    Test different aperture combinations for toxic and non-toxic conceptors.
    
    Args:
        args: Arguments object
        scores: Scores dictionary
        train_toxic_activations: Toxic training activations
        train_non_toxic_activations: Non-toxic training activations
        validation_set_activations: Validation set activations
        validation_dataset: Validation dataset
        seed: Seed number
    """
    # Set the apertures to test
    apertures_to_test = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    settings = get_optimal_settings(args.model_name, seed, "conceptor_based")
    module_name = settings['best_layer']
    if "," in module_name:
        module_name = module_name.split(", ")[-1].strip()

    highest_accuracy = 0
    best_aperture = 0  # Default fallback
    
    print(f"Testing apertures for module: {module_name}")
    
    for aperture in apertures_to_test:
        print(f"\nProcessing {args.experiment_type} for seed {seed} with aperture={aperture}")
        # Get the activations for this layer
        acts_toxic = train_toxic_activations[module_name]
        acts_non_toxic = train_non_toxic_activations[module_name]
        # Calculate toxic and non-toxic conceptors with different apertures
        C_toxic = compute_conceptor(acts_toxic, aperture)
        C_nontoxic = compute_conceptor(acts_non_toxic, aperture)
        # Calculate performance
        TP, FP, TN, FN = calculate_average_score(validation_set_activations[module_name],
                                                validation_dataset,
                                                C_toxic, 
                                                C_nontoxic)
        # Calculate metrics
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # Store the accuracy in the scores dictionary per aperture value
        scores['validation_set_score'][aperture] = accuracy
        # Track best aperture
        if accuracy > highest_accuracy:
            highest_accuracy = accuracy
            best_aperture = [aperture]
        elif accuracy == highest_accuracy:
            best_aperture.append(aperture)
        print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")
    
    print(f"\nBest aperture found:")
    print(f"  Aperture: {best_aperture}")
    print(f"  Accuracy: {highest_accuracy:.4f}")
    
    # Create results directory
    os.makedirs(f"results/seed_{seed}/{args.experiment_type}/{args.model_name}/raw_results", exist_ok=True)
    
    # Create plot after calculating scores for all apertures
    create_plots.accuracy_vs_aperture(scores, apertures_to_test, module_name, None, args.model_name, seed, "aperture_testing", args.experiment_type)
    filename = f"results/seed_{seed}/{args.experiment_type}/{args.model_name}/raw_results/module_{module_name}_aperture_testing.json"
    store_results(filename, scores, args.model_name, args.experiment_type, module_name, args.n_examples, best_aperture, seed)
    
    # Run zoom-in experiment with the best aperture
    print(f"\nStarting zoom-in experiment around best aperture: {best_aperture}")
    aperture_zoom_in(args, scores, best_aperture, train_toxic_activations, train_non_toxic_activations, validation_set_activations, validation_dataset, seed)

def layers(args, scores, train_toxic_activations, train_non_toxic_activations, validation_set_activations, validation_dataset, seed):
    module_names = list(train_toxic_activations.keys())
    # set the module to test (can be removed later, just for experiments)
    for module_name in module_names:
        scores['validation_set_score'][module_name] = {} # Initialize scores for each module
    aperture = 0.01 # take the first aperture since we are only testing layers
    best_module_name = ""
    highest_accuracy = 0
    for module_name in module_names:
        print(f"\nProcessing {args.experiment_type} for seed {seed} with module {module_name}")
        # Get the activations for this layer of all examples
        acts_toxic = train_toxic_activations[module_name]
        acts_non_toxic = train_non_toxic_activations[module_name]
            
        # Calculate toxic and non-toxic conceptors
        C_toxic = compute_conceptor(acts_toxic, aperture)
        C_nontoxic = compute_conceptor(acts_non_toxic, aperture)
        
        TP, FP, TN, FN = calculate_average_score(validation_set_activations[module_name],
                                        validation_dataset,
                                        C_toxic, 
                                        C_nontoxic)
            
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
    create_plots.accuracy_vs_layers(scores, module_names, args.aperture, args.model_name, seed)
    filename = f"results/seed_{seed}/{args.experiment_type}/{args.model_name}/raw_results/layer_testing_with_aperture_{aperture}.json"
    store_results(filename, scores, args.model_name, args.experiment_type, module_name, args.n_examples, args.aperture, seed)
    print(f"Best module: {best_module_name} with accuracy: {highest_accuracy:.4f}")
    
    update_optimal_settings(
        model_name=args.model_name,
        seed=seed,
        layer=best_module_name,
        aperture=None,
        accuracy=highest_accuracy,
        experiment_type="conceptor_layers",
        notes="Best performing layer for toxic classification"
        )
    
def confusion_matrix(args, train_toxic_activations, train_non_toxic_activations, test_set_activations, test_dataset, seed):
    # Load best_n_examples from JSON
    with open("results/mean_scores/n_examples/best_n_examples_per_model.json", "r") as f:
        best_n_dict = json.load(f)
    model_name = args.model_name.replace("/", "-")
    best_n = None
    if model_name in best_n_dict and "conceptor_examples" in best_n_dict[model_name]:
        best_n = best_n_dict[model_name]["conceptor_examples"].get("best_n_examples", None)
    if best_n is None:
        raise ValueError(f"No best_n_examples found for {model_name} in conceptor confusion_matrix.")
    best_n = int(best_n)

    settings = get_optimal_settings(args.model_name, seed, "conceptor_based")
    
    scores = {} # TP, FP, TN, FN, accuracy, precision, recall, f1_score
    aperture = settings['best_aperture']
    # Take the first aperture if multiple
    if isinstance(aperture, list):
        aperture = aperture[0]
    module_name = settings['best_layer']
    if "," in module_name:
        module_name = module_name.split(", ")[-1].strip()  # Take the last layer if multiple
    n_examples = best_n
    
    print(f"\nProcessing {args.experiment_type} for seed {seed}")
    print(f"Using aperture: {aperture}")
    print(f"Using n_examples: {n_examples}")
    print(f"Using module: {module_name}")
    
    # Get the activations for this layer of all examples
    acts_toxic = train_toxic_activations[module_name][:n_examples]
    acts_non_toxic = train_non_toxic_activations[module_name][:n_examples]
    
    # Calculate toxic and non-toxic conceptors with the same aperture
    C_toxic = compute_conceptor(acts_toxic, aperture)
    C_nontoxic = compute_conceptor(acts_non_toxic, aperture)
    
    TP, FP, TN, FN = calculate_average_score(test_set_activations[module_name],
                                    test_dataset,
                                    C_toxic, 
                                    C_nontoxic)
        
    # Calculate the average scores
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

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
    create_plots.confusion_matrix(TP, FP, TN, FN, aperture, args.model_name, module_name, seed, args.experiment_type)
    filename = f"results/seed_{seed}/{args.experiment_type}_for_best_n_examples/{args.model_name}/raw_results/module_{module_name}_with_aperture_{aperture}.json"
    store_results(filename, scores, args.model_name, args.experiment_type, module_name, n_examples, aperture, seed)


        
        
