## Datasets
*You have to run datasets/create_validation_split.py before running any experiments. This is done already in scripts/generate_steering_mechanism.py and scripts/ab_test_steering.py.*

This folder contains all raw and processed data used in the paper https://arxiv.org/abs/2312.06681. Their code and all following information in this README file about these datasets can also be found on their Github: https://github.com/nrimsky/CAA/tree/main?tab=readme-ov-file#datasets. The generate and test datasets were generated from the raw data through their https://github.com/nrimsky/CAA/blob/main/process_raw_datasets.py script. This function (briefly summarized) does the following:

- Loads raw JSON data for a given behavior and shuffles the data randomly into 50 items for the test set (found in the test directory) and a maximum of 1000 items for the generate set depending on how many total items are available (e.g., if there are 500 items in the raw data, 450 can be used for the generate data). It also generates an open-ended test file which can be judged with an llm. 

An overview of the data and where to find it (copied from the above repository):

### Coordination with other AIs (`coordinate-other-ais`)

Anthropic human generated eval data.

- https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/human_generated_evals/coordinate-other-ais.jsonl

### Corrigibility (`corrigible-neutral-HHH`)

Anthropic human generated eval data.

- https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/human_generated_evals/corrigible-neutral-HHH.jsonl

### Hallucination (`hallucination`)

Generated using GPT-4 by Wuschel Schulz [(source)](https://github.com/wusche1/CAA_hallucination/tree/main/paper/Hallucination/Datasets/HOCUS/questions)

### Myopia (`myopic-reward`)

Anthropic human generated eval data.

- https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/human_generated_evals/myopic-reward.jsonl

### Survival Instinct (`survival-instinct`)

Anthropic human generated eval data.

- https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/advanced-ai-risk/human_generated_evals/survival-instinct.jsonl

### Sycophancy (`sycophancy`)

Mixture of Anthropic's Sycophancy datasets.

- https://huggingface.co/datasets/Anthropic/model-written-evals/blob/main/sycophancy/

### Refusal (`refusal`)

Generated using GPT-4.

### TruthfulQA (`truthfulqa`) (_test only_)

- https://huggingface.co/datasets/truthful_qa/tree/main

### MMLU (`mmlu`) (_test only_)

`mmlu_full.json` is the full MMLU test dataset formatted as A/B questions. `mmlu.json` is a subset of $10$ questions from every category, which is what we use for evaluation.

- https://huggingface.co/datasets/cais/mmlu

## Final dataset sizes

```
coordinate-other-ais: n_generate: 360 | n_test: 50
corrigible-neutral-HHH: n_generate: 290 | n_test: 50
hallucination: n_generate: 1000 | n_test: 50
myopic-reward: n_generate: 950 | n_test: 50
survival-instinct: n_generate: 903 | n_test: 50
sycophancy: n_generate: 1000 | n_test: 50
refusal: n_generate: 408 | n_test: 50
```