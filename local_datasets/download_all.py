"""
Download all the datasets from the links below.

TODO:
- truthful_qa
- mmlu
- refusal
"""

import os
import requests

RAW_DATAFOLDER = "raw"

links = [
    "https://huggingface.co/datasets/Anthropic/model-written-evals/resolve/main/advanced-ai-risk/human_generated_evals/coordinate-other-ais.jsonl",
    "https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/human_generated_evals/corrigible-neutral-HHH.jsonl",
    "https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/human_generated_evals/myopic-reward.jsonl",
    "https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/human_generated_evals/survival-instinct.jsonl",
]

for url in links:
    filename = url.split("/")[-1]
    foldername = url.split("/")[-1].split(".")[0]
    foldername = os.path.join(RAW_DATAFOLDER, foldername)
    try:
        response = requests.get(url)
        response.raise_for_status()

        os.makedirs(foldername, exist_ok=True)

        with open(os.path.join(foldername, filename), "wb") as f:
            f.write(response.content)
        print(f"Successfully downloaded {url} to {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

sycophancy_links = [
    "https://huggingface.co/datasets/Anthropic/model-written-evals/resolve/main/sycophancy/sycophancy_on_nlp_survey.jsonl",
    "https://huggingface.co/datasets/Anthropic/model-written-evals/resolve/main/sycophancy/sycophancy_on_political_typology_quiz.jsonl",
]

for url in sycophancy_links:
    filename = url.split("/")[-1]
    foldername = "sycophancy"
    foldername = os.path.join(RAW_DATAFOLDER, foldername)
    try:
        response = requests.get(url)
        response.raise_for_status()

        os.makedirs(foldername, exist_ok=True)

        with open(os.path.join(foldername, filename), "wb") as f:
            f.write(response.content)
        print(f"Successfully downloaded {url} to {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")

hallucination_links = [
    "https://raw.githubusercontent.com/wusche1/CAA_hallucination/refs/heads/main/paper/Hallucination/Datasets/HOCUS/questions/alluding_questioning.csv",
    "https://raw.githubusercontent.com/wusche1/CAA_hallucination/refs/heads/main/paper/Hallucination/Datasets/HOCUS/questions/direct_questions.csv",
    "https://raw.githubusercontent.com/wusche1/CAA_hallucination/refs/heads/main/paper/Hallucination/Datasets/HOCUS/questions/questioning_assuming_statement.csv",
]

for url in hallucination_links:
    filename = url.split("/")[-1]
    foldername = "hallucination"
    foldername = os.path.join(RAW_DATAFOLDER, foldername)
    try:
        response = requests.get(url)
        response.raise_for_status()

        os.makedirs(foldername, exist_ok=True)

        with open(os.path.join(foldername, filename), "wb") as f:
            f.write(response.content)
        print(f"Successfully downloaded {url} to {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
