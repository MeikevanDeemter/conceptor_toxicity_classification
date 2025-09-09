import json
import os
from typing import Dict, Any, Optional
from datetime import datetime


def update_optimal_settings(model_name: str, seed: int, layer: str = None, aperture: float = None, accuracy: float = None, experiment_type: str = "conceptor_layers", notes: str = ""):
    """
    Update the optimal settings for a specific model and seed combination in the JSON file.
    
    Args:
        model_name: Name of the model (e.g., "EleutherAI/gpt-neo-2.7B")
        seed: Random seed used
        layer: Best performing layer (e.g., "transformer.h.31.ln_2") - only for layer experiments
        aperture: Best performing aperture value - only for conceptor aperture experiments
        accuracy: Accuracy achieved with these settings
        experiment_type: Type of experiment that produced these results
        notes: Additional notes about the settings
    """
    # Sanitize model name for consistency
    model_name = model_name.replace('/', '-')
    
    # Load existing settings from file
    file_settings = {}
    if os.path.exists("optimal_settings.json"):
        try:
            with open("optimal_settings.json", 'r') as f:
                file_settings = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading existing settings: {e}")
            file_settings = {}
    
    # Initialize model entry if it doesn't exist
    if model_name not in file_settings:
        file_settings[model_name] = {}
    
    # Convert seed to string for JSON compatibility
    seed_str = str(seed)
    
    # Initialize seed entry if it doesn't exist
    if seed_str not in file_settings[model_name]:
        file_settings[model_name][seed_str] = {
            "conceptor_based": {
                "best_layer": "unknown",
                "best_aperture": 0.01,
                "best_accuracy": 0.0,
                "experiment_type": "unknown",
                "notes": "",
                "last_updated": ""
            },
            "mean_activation_based": {
                "best_layer": "unknown",
                "best_accuracy": 0.0,
                "experiment_type": "unknown",
                "notes": "",
                "last_updated": ""
            }
        }
    
    # Determine which section to update based on experiment type
    if experiment_type.startswith("conceptor_"):
        section = "conceptor_based"
        section_name = "Conceptor-based"
    elif experiment_type.startswith("mean_activations_"):
        section = "mean_activation_based"
        section_name = "Mean activation-based"
    else:
        # Default to conceptor-based for backward compatibility
        section = "conceptor_based"
        section_name = "Conceptor-based"
    
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Update based on experiment type
    if experiment_type == "conceptor_layers" and layer is not None:
        # Update layer information from conceptor layer experiment
        file_settings[model_name][seed_str][section]["best_layer"] = layer
        if accuracy is not None:
            file_settings[model_name][seed_str][section]["best_accuracy"] = accuracy
        file_settings[model_name][seed_str][section]["experiment_type"] = experiment_type
        if notes:
            file_settings[model_name][seed_str][section]["notes"] = notes
        file_settings[model_name][seed_str][section]["last_updated"] = timestamp
            
        print(f"Updated optimal layer for {model_name}, seed {seed} ({section_name}):")
        print(f"  Layer: {layer}")
        print(f"  Accuracy: {accuracy:.4f}" if accuracy is not None else "  Accuracy: Not provided")
        
    elif experiment_type == "conceptor_aperture" and aperture is not None:
        # Update aperture information from conceptor aperture experiment
        file_settings[model_name][seed_str][section]["best_aperture"] = aperture
        if accuracy is not None:
            file_settings[model_name][seed_str][section]["best_accuracy"] = accuracy
        file_settings[model_name][seed_str][section]["experiment_type"] = experiment_type
        if notes:
            file_settings[model_name][seed_str][section]["notes"] = notes
        file_settings[model_name][seed_str][section]["last_updated"] = timestamp
            
        print(f"Updated optimal aperture for {model_name}, seed {seed} ({section_name}):")
        print(f"  Aperture: {aperture}")
        print(f"  Accuracy: {accuracy:.4f}" if accuracy is not None else "  Accuracy: Not provided")
        
    elif experiment_type == "conceptor_aperture_zoom_in" and aperture is not None:
        # Update aperture information from conceptor zoom-in experiment
        file_settings[model_name][seed_str][section]["best_aperture"] = aperture
        if accuracy is not None:
            file_settings[model_name][seed_str][section]["best_accuracy"] = accuracy
        file_settings[model_name][seed_str][section]["experiment_type"] = experiment_type
        if notes:
            file_settings[model_name][seed_str][section]["notes"] = notes
        file_settings[model_name][seed_str][section]["last_updated"] = timestamp
            
        print(f"Updated optimal aperture (zoom-in) for {model_name}, seed {seed} ({section_name}):")
        print(f"  Aperture: {aperture}")
        print(f"  Accuracy: {accuracy:.4f}" if accuracy is not None else "  Accuracy: Not provided")
        
    elif experiment_type == "mean_activations_layers" and layer is not None:
        # Update layer information from mean activations layer experiment
        file_settings[model_name][seed_str][section]["best_layer"] = layer
        if accuracy is not None:
            file_settings[model_name][seed_str][section]["best_accuracy"] = accuracy
        file_settings[model_name][seed_str][section]["experiment_type"] = experiment_type
        if notes:
            file_settings[model_name][seed_str][section]["notes"] = notes
        file_settings[model_name][seed_str][section]["last_updated"] = timestamp
            
        print(f"Updated optimal layer for {model_name}, seed {seed} ({section_name}):")
        print(f"  Layer: {layer}")
        print(f"  Accuracy: {accuracy:.4f}" if accuracy is not None else "  Accuracy: Not provided")
        
    else:
        # General update for other experiment types
        if layer is not None:
            file_settings[model_name][seed_str][section]["best_layer"] = layer
        if aperture is not None and section == "conceptor_based":
            file_settings[model_name][seed_str][section]["best_aperture"] = aperture
        if accuracy is not None:
            file_settings[model_name][seed_str][section]["best_accuracy"] = accuracy
        if experiment_type != "unknown":
            file_settings[model_name][seed_str][section]["experiment_type"] = experiment_type
        if notes:
            file_settings[model_name][seed_str][section]["notes"] = notes
        file_settings[model_name][seed_str][section]["last_updated"] = timestamp
            
        print(f"Updated optimal settings for {model_name}, seed {seed} ({section_name}):")
        if layer is not None:
            print(f"  Layer: {layer}")
        if aperture is not None and section == "conceptor_based":
            print(f"  Aperture: {aperture}")
        if accuracy is not None:
            print(f"  Accuracy: {accuracy:.4f}")
    
    # Save the updated settings to file
    save_settings_to_file(file_settings)

def get_optimal_settings(model_name: str, seed: int, classification_type: str = "conceptor_based") -> Optional[Dict[str, Any]]:
    """
    Get the optimal settings for a specific model and seed combination from the JSON file.
    
    Args:
        model_name: Name of the model
        seed: Random seed used
        classification_type: Either "conceptor_based" or "mean_activation_based"
        
    Returns:
        Dictionary with optimal settings or None if not found
    """
    model_name = model_name.replace('/', '-')
    
    # Load settings from file
    if os.path.exists("optimal_settings.json"):
        try:
            with open("optimal_settings.json", 'r') as f:
                file_settings = json.load(f)
            
            # Convert seed to string since JSON stores keys as strings
            seed_str = str(seed)
            
            if model_name in file_settings and seed_str in file_settings[model_name]:
                # Check if the new structure exists, otherwise return the old structure for backward compatibility
                if classification_type in file_settings[model_name][seed_str]:
                    return file_settings[model_name][seed_str][classification_type]
                else:
                    # Backward compatibility: return the old structure
                    return file_settings[model_name][seed_str]
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading settings from file: {e}")
            return None
    else:
        print("optimal_settings.json file not found")
        return None
    
    return None

def print_all_settings():
    """Print all stored optimal settings in a formatted way."""
    print("Optimal Settings Summary:")
    print("=" * 80)
    
    if not os.path.exists("optimal_settings.json"):
        print("No optimal_settings.json file found.")
        return
        
    try:
        with open("optimal_settings.json", 'r') as f:
            file_settings = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Error loading settings: {e}")
        return
    
    for model_name, seeds in file_settings.items():
        print(f"\nModel: {model_name}")
        print("-" * 40)
        
        for seed, settings in seeds.items():
            print(f"  Seed {seed}:")
            
            # Check if new structure exists
            if "conceptor_based" in settings and "mean_activation_based" in settings:
                # New structure
                print(f"    Conceptor-based classification:")
                conceptor = settings["conceptor_based"]
                print(f"      Layer: {conceptor['best_layer']}")
                print(f"      Aperture: {conceptor['best_aperture']}")
                accuracy = conceptor['best_accuracy']
                if isinstance(accuracy, (int, float)) and accuracy != 0:
                    print(f"      Accuracy: {accuracy:.4f}")
                else:
                    print(f"      Accuracy: {accuracy}")
                print(f"      Experiment: {conceptor['experiment_type']}")
                if conceptor.get('notes'):
                    print(f"      Notes: {conceptor['notes']}")
                if conceptor.get('last_updated'):
                    print(f"      Last updated: {conceptor['last_updated']}")
                
                print(f"    Mean activation-based classification:")
                mean_act = settings["mean_activation_based"]
                print(f"      Layer: {mean_act['best_layer']}")
                accuracy = mean_act['best_accuracy']
                if isinstance(accuracy, (int, float)) and accuracy != 0:
                    print(f"      Accuracy: {accuracy:.4f}")
                else:
                    print(f"      Accuracy: {accuracy}")
                print(f"      Experiment: {mean_act['experiment_type']}")
                if mean_act.get('notes'):
                    print(f"      Notes: {mean_act['notes']}")
                if mean_act.get('last_updated'):
                    print(f"      Last updated: {mean_act['last_updated']}")
            else:
                # Old structure (backward compatibility)
                print(f"      Layer: {settings['best_layer']}")
                print(f"      Aperture: {settings['best_aperture']}")
                accuracy = settings['best_accuracy']
                if isinstance(accuracy, (int, float)) and accuracy != 0:
                    print(f"      Accuracy: {accuracy:.4f}")
                else:
                    print(f"      Accuracy: {accuracy}")
                print(f"      Experiment: {settings['experiment_type']}")
                if settings.get('notes'):
                    print(f"      Notes: {settings['notes']}")
            print()

def save_settings_to_file(file_settings, filename: str = "optimal_settings.json"):
    """Save the optimal settings dictionary to optimal_settings.json."""
    with open(filename, 'w') as f:
        json.dump(file_settings, f, indent=2)
    print(f"Optimal settings saved to {filename}")

def load_settings_from_file(filename: str = "optimal_settings.json"):
    """Load the optimal settings dictionary from a JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    else:
        print(f"File {filename} not found. Using default settings.")
        return {}

def add_to_results(best_module_name, highest_accuracy, model_name, seed):
    # append the setting to the settings.json file (aperture, model, seed) based
    # on experiment type
    if experiment_type == "conceptor_layers":
        settings = {
            "layer": best_module_name,
            "model": model_name,
            "seed": seed
        }
    elif experiment_type == "conceptor_aperture_zoom_in":
        settings = {
            "aperture": aperture,
            "model": model_name,
            "seed": seed
        }
    elif experiment_type == "mean_activations_layers":
        settings = {
            "layer": best_module_name,
            "model": model_name,
            "seed": seed
        }
    
    # Use the new update function
    update_optimal_settings(
        model_name=model_name,
        seed=seed,
        layer=best_module_name if 'layer' in locals() else "unknown",
        aperture=0.01,  # Default aperture, should be updated based on experiment
        accuracy=highest_accuracy,
        experiment_type=experiment_type if 'experiment_type' in locals() else "unknown"
    )

# Example usage functions
def example_usage():
    """Example of how to use the optimal settings system."""
    
    # Update settings for a specific model and seed
    update_optimal_settings(
        model_name="EleutherAI/gpt-neo-2.7B",
        seed=42,
        layer="transformer.h.31.ln_2",
        aperture=0.01,
        accuracy=0.8542,
        notes="Best performing layer for toxic classification"
    )
    
    # Get settings for a specific model and seed
    settings = get_optimal_settings("EleutherAI/gpt-neo-2.7B", 42)
    if settings:
        print(f"Best layer: {settings['best_layer']}")
        print(f"Best aperture: {settings['best_aperture']}")
        print(f"Best accuracy: {settings['best_accuracy']}")
    
    # Save settings to file
    save_settings_to_file(settings)
    
    # Print all settings
    print_all_settings()

if __name__ == "__main__":
    # Load existing settings if available
    settings = load_settings_from_file()
    
    # Print current settings
    print_all_settings()



    