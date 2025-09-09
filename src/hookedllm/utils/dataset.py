"""Dataset handling utilities."""

from typing import List, Dict, Optional
import json
from pathlib import Path


def get_examples(dataset_name: str, generate: bool = True, match: bool = True, base_path: Optional[Path] = None) -> List[str]:
    """Load examples from the specified dataset.
    
    Args:
    ----
        dataset_name: Name of the dataset to load
        generate: Whether to load from generate or test dataset
        match: Whether to get matching or non-matching behavior examples
        base_path: Base path for datasets, defaults to repository root
        
    Returns:
    -------
        List of example strings from the dataset
    """
    if base_path is None:
        base_path = Path(__file__).parents[3]
    
    if generate:
        dataset_path = base_path / "datasets" / "generate" / dataset_name / "generate_dataset.json"
    else:
        dataset_path = base_path / "datasets" / "test" / dataset_name / "test_dataset_ab.json"
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # For the particular dataset, we want to get the examples with matching or non-matching behavior depending on the match flag
    examples = []
    if match:
        for item in data:
            example = f"{item['question']}\n{item['answer_matching_behavior']}"
            examples.append(example)
    else:
        for item in data:
            example = f"{item['question']}\n{item['answer_not_matching_behavior']}"
            examples.append(example)
    
    return examples


def get_test_prompts(dataset_name: str, base_path: Optional[Path] = None) -> List[Dict]:
    """Load test prompts without answers, but track both answer types for evaluation.
    
    Args:
    ----
        dataset_name: Name of the dataset to load
        base_path: Base path for datasets, defaults to repository root
        
    Returns:
    -------
        List of prompt dictionaries with question and expected answers
    """
    if base_path is None:
        base_path = Path(__file__).parents[3]
        
    dataset_path = base_path / "datasets" / "test" / dataset_name / "test_dataset_ab.json"
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    prompts = []
    
    # Print a sample for verification
    if len(data) > 0:
        sample_item = data[0]
        bad_answer = sample_item["answer_matching_behavior"]
        good_answer = sample_item["answer_not_matching_behavior"]
        
        print("\n=== DATASET ANSWER PATTERN ===")
        print(f"Good behavior answer (answer_not_matching_behavior): {good_answer[-3:]} - {good_answer[:50]}...")
        print(f"Bad behavior answer (answer_matching_behavior): {bad_answer[-3:]} - {bad_answer[:50]}...")
    
    # Process all items
    for item in data:
        good_answer = item["answer_not_matching_behavior"][-3:] if len(item["answer_not_matching_behavior"]) >= 3 else None
        bad_answer = item["answer_matching_behavior"][-3:] if len(item["answer_matching_behavior"]) >= 3 else None
        
        prompts.append({
            "question": item["question"],
            "good_behavior_answer": good_answer,  # The answer we want the model to produce
            "bad_behavior_answer": bad_answer     # The answer we want to steer away from
        })
    
    return prompts


def get_open_ended_test_prompts(dataset_name: str, base_path: Optional[Path] = None) -> List[Dict]:
    """Load open-ended test prompts.
    
    Args:
    ----
        dataset_name: Name of the dataset to load
        base_path: Base path for datasets, defaults to repository root
        
    Returns:
    -------
        List of prompt dictionaries with questions
    """
    if base_path is None:
        base_path = Path(__file__).parents[3]
        
    dataset_path = base_path / "datasets" / "test" / dataset_name / "test_dataset_open_ended.json"
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    prompts = []
    
    # Process all items
    for item in data:
        prompts.append({
            "question": item["question"]
        })
    
    return prompts 