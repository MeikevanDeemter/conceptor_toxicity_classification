from datasets import load_dataset
from pathlib import Path
from datasets import concatenate_datasets
import json
from transformers import AutoTokenizer
import os

def max_tokens(dataset):
    # Make sure the examples are not longer than 200 tokens, using a tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    dataset = dataset.filter(lambda x: len(tokenizer.encode(x['comment_text'])) <= 200)
    return dataset

def create_balanced_datasets(dataset, n_examples):
    # Create datasets of n_examples of toxic and non-toxic examples
    toxic_dataset = dataset.filter(lambda x: x['toxic'] == 1)
    non_toxic_dataset = dataset.filter(lambda x: x['toxic'] == 0)
    toxic_dataset = toxic_dataset.select(range(n_examples))
    non_toxic_dataset = non_toxic_dataset.select(range(n_examples))
    return toxic_dataset, non_toxic_dataset

def filter_dataset(dataset):
    # Filter dataset to only include entries where toxic is 0 or 1
    filtered_dataset = dataset.filter(lambda x: x['toxic'] != -1)
    
    # Filter dataset that only show signs of toxicity, not other categories
    filtered_dataset = filtered_dataset.filter(lambda x: x['severe_toxic'] == -1 or x['severe_toxic'] == 0)
    filtered_dataset = filtered_dataset.filter(lambda x: x['obscene'] == -1 or x['obscene'] == 0)
    filtered_dataset = filtered_dataset.filter(lambda x: x['threat'] == -1 or x['threat'] == 0)
    filtered_dataset = filtered_dataset.filter(lambda x: x['insult'] == -1 or x['insult'] == 0)
    filtered_dataset = filtered_dataset.filter(lambda x: x['identity_hate'] == -1 or x['identity_hate'] == 0)
    
    return filtered_dataset

def convert_and_save_to_json(dataset, output_path):
    # Convert dataset to list of dictionaries with text and labels
    json_data = []
    for item in dataset:
        json_data.append({
            "text": item["comment_text"],
            "labels": {
                "toxicity": float(item["toxic"]),
                "severe_toxicity": float(item["severe_toxic"]),
                "obscene": float(item["obscene"]),
                "threat": float(item["threat"]),
                "insult": float(item["insult"]),
                "identity_hate": float(item["identity_hate"])
            }
        })
    
    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)
    
    return 

def main():
    # Get the absolute path to the project root directory
    project_root = Path(__file__).parent.parent
    
    # Load the Jigsaw datasets
    jigsaw_data = load_dataset(
        "csv",
        data_dir=str(project_root / "datasets" / "jigsaw-data"),
        data_files={
            "train": "train-jigsaw-toxic-data.csv",
            "test": "test-with-labels-jigsaw-toxic-data.csv",
        }
    )
    
    # Filter datasets so they only contain entries where toxic is 0 or 1 
    # where other categories are either 0 or -1 (so they are not signs of that category)
    train_dataset = filter_dataset(jigsaw_data['train'])
    test_dataset = filter_dataset(jigsaw_data['test'])
    
    # Make sure the examples are not longer than 200 tokens
    print("Maximizing token length...")
    train_dataset = max_tokens(train_dataset)
    test_dataset = max_tokens(test_dataset)

    # Create validation set from training set (20% of training set)
    train_val = train_dataset.train_test_split(test_size=0.2)
    train_dataset = train_val['train']
    validation_dataset = train_val['test']
    
    toxic_train_dataset = train_dataset.filter(lambda x: x['toxic'] == 1)
    non_toxic_train_dataset = train_dataset.filter(lambda x: x['toxic'] == 0)
    toxic_validation_dataset = validation_dataset.filter(lambda x: x['toxic'] == 1)
    non_toxic_validation_dataset = validation_dataset.filter(lambda x: x['toxic'] == 0)
    toxic_test_dataset = test_dataset.filter(lambda x: x['toxic'] == 1)
    non_toxic_test_dataset = test_dataset.filter(lambda x: x['toxic'] == 0)
    
    print("Total number of toxic training examples: ", len(toxic_train_dataset))
    print("Total number of non-toxic training examples: ", len(non_toxic_train_dataset))
    print("Total number of toxic validation examples: ", len(toxic_validation_dataset))
    print("Total number of non-toxic validation examples: ", len(non_toxic_validation_dataset))
    print("Total number of toxic test examples: ", len(toxic_test_dataset))
    print("Total number of non-toxic test examples: ", len(non_toxic_test_dataset))
    
    breakpoint()
    
    # Shuffle the datasets with 10 different seeds = [42, 1337, 2024, 7, 8675309, 31415, 12345, 777, 9001, 271828]
    seeds = [42, 1337, 2024, 7, 8675309, 31415, 12345, 777, 9001, 271828]
    # seeds = [42, 1337, 2024, 7, 8675309]
    for seed in seeds:
        # Create fresh copies of the datasets for each seed
        train_dataset_seed = train_dataset.shuffle(seed=seed)
        validation_dataset_seed = validation_dataset.shuffle(seed=seed)
        test_dataset_seed = test_dataset.shuffle(seed=seed)

        # Create balanced datasets with n_examples of toxic and non-toxic examples
        print("Creating balanced datasets...")
        train_dataset_toxic, train_dataset_non_toxic = create_balanced_datasets(train_dataset_seed, 3000)
        validation_dataset_toxic, validation_dataset_non_toxic = create_balanced_datasets(validation_dataset_seed, 375)
        test_dataset_toxic, test_dataset_non_toxic = create_balanced_datasets(test_dataset_seed, 375)
    
        # Combine and shuffle validation and testing datasets
        print("Combining and shuffling toxic and non-toxic examples in validation and testing datasets...")
        validation_dataset_final = concatenate_datasets([validation_dataset_toxic, validation_dataset_non_toxic])
        validation_dataset_final = validation_dataset_final.shuffle(seed=42)
        test_dataset_final = concatenate_datasets([test_dataset_toxic, test_dataset_non_toxic])
        test_dataset_final = test_dataset_final.shuffle(seed=42)
        train_dataset_final = concatenate_datasets([train_dataset_toxic, train_dataset_non_toxic])
    
    
        # Convert datasets to JSON format and save to files
        print("Converting and saving datasets...")
        os.makedirs(f"{project_root}/datasets/seed_{seed}", exist_ok=True)
        convert_and_save_to_json(train_dataset_final, f"{project_root}/datasets/seed_{seed}/train_dataset.json")
        convert_and_save_to_json(train_dataset_toxic, f"{project_root}/datasets/seed_{seed}/train_dataset_toxic.json")
        convert_and_save_to_json(train_dataset_non_toxic, f"{project_root}/datasets/seed_{seed}/train_dataset_non_toxic.json")
        convert_and_save_to_json(validation_dataset_final, f"{project_root}/datasets/seed_{seed}/validation_dataset.json")
        convert_and_save_to_json(test_dataset_final, f"{project_root}/datasets/seed_{seed}/test_dataset.json")
        
        print("Total number of training examples: ", len(train_dataset_final))
        print("Total number of toxic training examples: ", len(train_dataset_toxic))
        print("Total number of non-toxic training examples: ", len(train_dataset_non_toxic))
        print("Total number of validation examples: ", len(validation_dataset_final))
        print("Total number of test examples: ", len(test_dataset_final))
        print(f"Balanced datasets (50/50) created successfully for seed {seed}!")

if __name__ == "__main__":
    main()