import os
import json
import random
import sys

# Constants
N_TEST = 50
N_GENERATE = 150  # Fixed number of samples for training
# The left overs will be used for validation
SEED = 42  # Fixed seed for reproducibility

def create_dataset_splits(verbose=True):
    """
    Create generate, validation, and test splits from raw data.
    
    Args:
        verbose: Whether to print detailed progress information
    
    Returns:
        List of behavior names for which datasets were successfully created
    """
    # Set random seed for reproducibility
    random.seed(SEED)
    
    # Define base paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if verbose:
        print(f"Base directory: {base_dir}")
    
    raw_dir = os.path.join(base_dir, "raw")
    generate_dir = os.path.join(base_dir, "generate")
    test_dir = os.path.join(base_dir, "test")
    validation_dir = os.path.join(base_dir, "validation")
    
    if verbose:
        print(f"Raw directory: {raw_dir}")
        print(f"Generate directory: {generate_dir}")
        print(f"Test directory: {test_dir}")
        print(f"Validation directory: {validation_dir}")
    
    # Check if raw directory exists
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory {raw_dir} does not exist!")
        return []
    
    # Get all behavior folders from raw directory
    behaviors = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
    if verbose:
        print(f"Found behaviors: {behaviors}")
    
    successful_behaviors = []
    
    for behavior in behaviors:
        # Define paths for this behavior
        raw_data_path = os.path.join(raw_dir, behavior, f"dataset.json")
        if verbose:
            print(f"Looking for raw data at: {raw_data_path}")
        
        # Skip if raw data file doesn't exist
        if not os.path.exists(raw_data_path):
            if verbose:
                print(f"Warning: {raw_data_path} not found, skipping {behavior}")
            continue
        
        # Create directories if they don't exist
        generate_behavior_dir = os.path.join(generate_dir, behavior)
        test_behavior_dir = os.path.join(test_dir, behavior)
        validation_behavior_dir = os.path.join(validation_dir, behavior)
        
        os.makedirs(generate_behavior_dir, exist_ok=True)
        os.makedirs(test_behavior_dir, exist_ok=True)
        os.makedirs(validation_behavior_dir, exist_ok=True)
        
        if verbose:
            print(f"Created directories for {behavior}")
        
        # Define output paths
        generate_path = os.path.join(generate_behavior_dir, "generate_dataset.json")
        test_ab_path = os.path.join(test_behavior_dir, "test_dataset_ab.json")
        test_open_ended_path = os.path.join(test_behavior_dir, "test_dataset_open_ended.json")
        validation_ab_path = os.path.join(validation_behavior_dir, "validation_dataset_ab.json")
        validation_open_ended_path = os.path.join(validation_behavior_dir, "validation_dataset_open_ended.json")
        
        if verbose:
            print(f"Loading raw data from {raw_data_path}")
        # Load raw data
        try:
            with open(raw_data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if verbose:
                print(f"Loaded {len(data)} items from raw data")
        except Exception as e:
            print(f"Error loading raw data: {e}")
            continue
        
        # Clean and format data
        cleaned_data = []
        for item in data:
            try:
                question = item["question"]
                answer_matching_behavior = item["answer_matching_behavior"]
                answer_not_matching_behavior = item["answer_not_matching_behavior"]
                
                # Handle if answers are lists
                if isinstance(answer_matching_behavior, list):
                    answer_matching_behavior = answer_matching_behavior[0]
                if isinstance(answer_not_matching_behavior, list):
                    answer_not_matching_behavior = answer_not_matching_behavior[0]
                
                cleaned_data.append({
                    "question": question.replace("Question:", "").replace("Answer:", "").strip(),
                    "answer_matching_behavior": answer_matching_behavior.strip(),
                    "answer_not_matching_behavior": answer_not_matching_behavior.strip(),
                })
            except Exception as e:
                if verbose:
                    print(f"Error processing item: {e}")
                continue
        
        if verbose:
            print(f"Cleaned {len(cleaned_data)} items")
        
        # Shuffle data with fixed seed
        random.shuffle(cleaned_data)
        
        # Calculate split sizes - use as many samples as available
        total_data = len(cleaned_data)
        
        # Ensure we have minimum data required
        if total_data < N_GENERATE + N_TEST:
            print(f"Not enough data for {behavior}. Total data: {total_data}, need at least {N_GENERATE + N_TEST}")
            continue
        
        # Split data: exactly N_GENERATE for training, N_TEST for testing, and the rest for validation
        generate_data = cleaned_data[:N_GENERATE]
        test_data = cleaned_data[-N_TEST:]
        validation_data = cleaned_data[N_GENERATE:-N_TEST]  # All data between training and testing
        
        n_validation = len(validation_data)
        
        if verbose:
            print(f"Split data: generate={len(generate_data)}, validation={n_validation}, test={len(test_data)}")
        
        # Write generate data
        try:
            with open(generate_path, "w", encoding="utf-8") as f:
                json.dump(generate_data, f, indent=4, ensure_ascii=False)
            if verbose:
                print(f"Wrote generate data to {generate_path}")
        except Exception as e:
            print(f"Error writing generate data: {e}")
            continue
        
        # Write test data (AB format)
        try:
            with open(test_ab_path, "w", encoding="utf-8") as f:
                json.dump(test_data, f, indent=4, ensure_ascii=False)
            if verbose:
                print(f"Wrote test AB data to {test_ab_path}")
        except Exception as e:
            print(f"Error writing test AB data: {e}")
            continue
        
        # Write validation data (AB format)
        try:
            with open(validation_ab_path, "w", encoding="utf-8") as f:
                json.dump(validation_data, f, indent=4, ensure_ascii=False)
            if verbose:
                print(f"Wrote validation AB data to {validation_ab_path}")
        except Exception as e:
            print(f"Error writing validation AB data: {e}")
            continue
        
        # Create open-ended test data
        open_ended_test = []
        for item in test_data:
            question = item["question"]
            # Extract the question part (before choices if they exist)
            if "\n\nChoices:" in question:
                question = question.split("\n\nChoices:")[0].strip()
            else:
                question = question.split("\n")[0].strip()
                
            open_ended_test.append({"question": question})
        
        # Create open-ended validation data
        open_ended_validation = []
        for item in validation_data:
            question = item["question"]
            # Extract the question part (before choices if they exist)
            if "\n\nChoices:" in question:
                question = question.split("\n\nChoices:")[0].strip()
            else:
                question = question.split("\n")[0].strip()
                
            open_ended_validation.append({"question": question})
        
        # Write open-ended test data
        try:
            with open(test_open_ended_path, "w", encoding="utf-8") as f:
                json.dump(open_ended_test, f, indent=4, ensure_ascii=False)
            if verbose:
                print(f"Wrote test open-ended data to {test_open_ended_path}")
        except Exception as e:
            print(f"Error writing test open-ended data: {e}")
            continue
        
        # Write open-ended validation data
        try:
            with open(validation_open_ended_path, "w", encoding="utf-8") as f:
                json.dump(open_ended_validation, f, indent=4, ensure_ascii=False)
            if verbose:
                print(f"Wrote validation open-ended data to {validation_open_ended_path}")
        except Exception as e:
            print(f"Error writing validation open-ended data: {e}")
            continue
        
        print(f"{behavior}: n_generate: {N_GENERATE} | n_validation: {n_validation} | n_test: {N_TEST}")
        successful_behaviors.append(behavior)
    
    return successful_behaviors

if __name__ == "__main__":
    create_dataset_splits()