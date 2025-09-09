"""Function vectors data using HuggingFace datasets library."""

import json
import os
from typing import Optional, Union

from datasets import Dataset, DatasetDict


def load_function_vectors_dataset(
    data_path: str,
    test_size: Optional[float] = None,
    seed: Optional[int] = None,
) -> Union[Dataset, DatasetDict]:
    """Load function vectors data from JSON file into a HuggingFace Dataset.

    Args:
    ----
        data_path: Path to the JSON file containing function vectors data
        test_size: If provided, split into train/test sets with this ratio (optional)
        seed: Random seed for splitting (optional)

    Returns:
    -------
        HuggingFace Dataset or DatasetDict (if split)

    Raises:
    ------
        FileNotFoundError: If the data file is not found
        ValueError: If the data format is incorrect

    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Read JSON data
    with open(data_path, "r") as f:
        json_data = json.load(f)

    # Validate data format
    if not isinstance(json_data, list):
        raise ValueError(f"Expected a list of items, got {type(json_data)}")

    # Check if items have the expected format
    for item in json_data:
        if not isinstance(item, dict) or "input" not in item or "output" not in item:
            raise ValueError(f"Expected items with 'input' and 'output' keys, got {item}")

    # Create HuggingFace Dataset
    dataset = Dataset.from_list(json_data)

    # Split if requested
    if test_size is not None:
        dataset = dataset.train_test_split(test_size=test_size, seed=seed)

    return dataset


def get_antonyms_dataset(
    data_path: Optional[str] = None, test_size: Optional[float] = None, seed: Optional[int] = None
) -> Union[Dataset, DatasetDict]:
    """Load the antonyms dataset into a HuggingFace Dataset.

    Args:
    ----
        data_path: Path to the antonyms JSON file (defaults to standard location)
        test_size: If provided, split into train/test sets with this ratio (optional)
        seed: Random seed for splitting (optional)

    Returns:
    -------
        HuggingFace Dataset or DatasetDict (if split)

    """
    if data_path is None:
        # Try to locate the data file relative to the current file
        module_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(module_dir, "../.."))
        data_path = os.path.join(repo_root, "conceptor-steering/data/functionvectors/antonyms.json")

    return load_function_vectors_dataset(data_path=data_path, test_size=test_size, seed=seed)


def create_few_shot_dataset(
    dataset: Dataset,
    num_examples: int = 3,
    example_format: str = "{input}:{output}",
    query_format: str = "{input}:",
    delimiter: str = ", ",
    seed: Optional[int] = None,
) -> Dataset:
    """Create a dataset with few-shot prompts.

    Args:
    ----
        dataset: HuggingFace Dataset with function vectors data
        num_examples: Number of examples to include per prompt
        example_format: Format string for examples
        query_format: Format string for the query
        delimiter: Delimiter for joining examples
        seed: Random seed for example selection

    Returns:
    -------
        Dataset with fields:
            'prompt' (input:output,...)
            '_query' (last input)
            'answer' (last output, not contained in prompt)

    """
    # Shuffle dataset if seed is provided
    if seed is not None:
        dataset = dataset.shuffle(seed=seed)

    prompts = []

    # Create a prompt for each item
    for i, item in enumerate(dataset):
        # Get examples (excluding the current item)
        example_indices = list(range(len(dataset)))
        example_indices.remove(i)

        # Select random examples
        if len(example_indices) > num_examples:
            if seed is not None:
                # Deterministic shuffling based on item index and seed
                import random

                r = random.Random(seed + i)
                r.shuffle(example_indices)
                example_indices = example_indices[:num_examples]
            else:
                example_indices = example_indices[i : i + num_examples]

        examples = [dataset[idx] for idx in example_indices]

        # Format the examples
        formatted_examples = []
        for example in examples:
            formatted_examples.append(
                example_format.format(input=example["input"], output=example["output"])
            )

        # Join examples and add query
        examples_text = delimiter.join(formatted_examples)
        query = query_format.format(input=item["input"])

        if examples_text:
            prompt = f"{examples_text}{delimiter}{query}"
        else:
            prompt = query

        prompts.append({"prompt": prompt, "_query": item["input"], "answer": item["output"]})

    return Dataset.from_list(prompts)


def get_function_vectors_data_loader(
    data_path: Optional[str] = None,
    test_size: Optional[float] = None,
    seed: Optional[int] = None,
):
    """Get a function vectors dataset and dataloader.

    Args:
    ----
        data_path: Path to the antonyms JSON file (defaults to standard location)
        test_size: If provided, split into train/test sets with this ratio (optional)
        seed: Random seed for splitting (optional)

    Returns:
    -------
        HuggingFace Dataset or DatasetDict (if split)

    """
    if data_path is None:
        # Try to locate the data file relative to the current file
        module_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(module_dir, "../.."))
        data_path = os.path.join(repo_root, "conceptor-steering/data/functionvectors/antonyms.json")

    return load_function_vectors_dataset(data_path=data_path, test_size=test_size, seed=seed)


def filter_by_length(
    dataset: Dataset,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    column: str = "input",
) -> Dataset:
    """Filter dataset by text length.

    Args:
    ----
        dataset: HuggingFace Dataset to filter
        min_length: Minimum text length (inclusive)
        max_length: Maximum text length (inclusive)
        column: Column to check length

    Returns:
    -------
        Filtered dataset

    """

    def length_filter(example):
        text_len = len(example[column])
        if min_length is not None and text_len < min_length:
            return False
        if max_length is not None and text_len > max_length:
            return False
        return True

    return dataset.filter(length_filter)
