"""Evaluation utilities for measuring steering performance."""

from typing import Dict, List, Tuple, Any, Optional
import torch

from hookedllm.steering_functions import SteeringFunction


# ANSI escape codes - imported in __init__.py
BLUE = '\033[94m'
RESET = '\033[0m'


def evaluate_prompts_with_logits(
    model, 
    prompts: List[Dict], 
    steering_fn: SteeringFunction, 
    hyperparam_strl: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Evaluate prompts by comparing logits for A/B answers and record results.
    
    Args:
    ----
        model: Model to evaluate
        prompts: List of prompt dictionaries with question and expected answers
        steering_fn: Steering function to apply
        hyperparam_strl: List of hyperparameter strings for result keys
        
    Returns:
    -------
        Dictionary of evaluation results
    """
    results = {}
    
    # --- Robust Token ID Fetching --- 
    token_ids = {}
    versions = {
        "space": [" A", " B"],
        "no_space": ["A", "B"]
    }
    
    # Try fetching all versions
    for space_type, (a_tok, b_tok) in versions.items():
        try:
            token_ids[f'a_{space_type}'] = model.tokenizer.convert_tokens_to_ids(a_tok)
            token_ids[f'b_{space_type}'] = model.tokenizer.convert_tokens_to_ids(b_tok)
            print(f"Found IDs for {space_type}: A={token_ids[f'a_{space_type}']}, B={token_ids[f'b_{space_type}']}")
        except KeyError:
            print(f"Warning: Could not get token IDs for {space_type} ('{a_tok}', '{b_tok}')")
            token_ids[f'a_{space_type}'] = None # Mark as not found
            token_ids[f'b_{space_type}'] = None

    # Ensure we found at least one version for A and B
    if (token_ids['a_space'] is None and token_ids['a_no_space'] is None) or \
       (token_ids['b_space'] is None and token_ids['b_no_space'] is None):
           raise ValueError("Could not find valid token IDs for A or B (both space/no-space versions)")
    # --- End Token ID Fetching ---

    for idx, prompt_data in enumerate(prompts):
        prompt = prompt_data["question"]
        good_behavior_answer = prompt_data["good_behavior_answer"]
        bad_behavior_answer = prompt_data["bad_behavior_answer"]
        
        # Format prompt exactly like CAA base model format
        eval_prompt = f"Input: {prompt.strip()}\nResponse: (" 
        
        # Run model.generate with steering to get scores for the next token
        with model.steering(steering_fn, steering_fn.config):
            # Generate only the next token and get its scores (logits)
            gen_kwargs = {
                "max_new_tokens": 1,
                "return_dict_in_generate": True,
                "output_scores": True,
                "pad_token_id": model.tokenizer.eos_token_id # Prevent warning
            }
            outputs = model.generate(eval_prompt, generation_kwargs=gen_kwargs)

        # Get logits for the first (and only) generated token
        logits = outputs.scores[0]
        # Logits shape is typically [batch_size, vocab_size]
        last_token_logits = logits[0, :] # Assuming batch size is 1

        # Calculate probabilities using softmax
        probs = torch.softmax(last_token_logits, dim=-1)
        
        # --- Sum Probabilities for A/B (Space and No-Space) --- 
        a_prob_total = 0.0
        b_prob_total = 0.0

        # Add probability if the token ID was found
        if token_ids['a_space'] is not None:
            a_prob_total += probs[token_ids['a_space']].item()
        if token_ids['a_no_space'] is not None:
             a_prob_total += probs[token_ids['a_no_space']].item()

        if token_ids['b_space'] is not None:
            b_prob_total += probs[token_ids['b_space']].item()
        if token_ids['b_no_space'] is not None:
            b_prob_total += probs[token_ids['b_no_space']].item()
        # --- End Sum Probabilities --- 
        
        # Determine predicted answer based on higher total probability
        predicted_answer = "(A)" if a_prob_total > b_prob_total else "(B)"
        
        # Check correctness against the original (A) / (B) format from dataset
        is_correct = predicted_answer == good_behavior_answer
        matches_good = "✓" if is_correct else "✗"
        matches_bad = predicted_answer == bad_behavior_answer
        matches_bad_str = "✓" if matches_bad else "✗"
        
        # Use eval_prompt in the print statement for clarity
        print(f"Prompt {idx+1}: {BLUE}{eval_prompt}{RESET}") 
        # Print the predicted answer in (A)/(B) format consistent with comparison
        # Show the summed probabilities
        print(f"Predicted: {predicted_answer} (A_prob_total={a_prob_total:.4f}, B_prob_total={b_prob_total:.4f})") 
        print(f"Matches good behavior {good_behavior_answer}: {matches_good}  |  Matches bad behavior {bad_behavior_answer}: {matches_bad_str}")

        # Determine result key
        if hyperparam_strl:
            result_key = f"{'_'.join(hyperparam_strl)}_{idx}"
        else:
            result_key = f"prompt_{idx}"

        # Store results
        results[result_key] = {
            "prompt": eval_prompt,
            "good_behavior_answer": good_behavior_answer,
            "bad_behavior_answer": bad_behavior_answer,
            "predicted_answer": predicted_answer,
            "a_prob": a_prob_total,
            "b_prob": b_prob_total,
            "matches_good_behavior": is_correct,
            "matches_bad_behavior": matches_bad
        }
        
    return results


def calculate_accuracy(results: Dict) -> Dict:
    """Calculate accuracy of the model's predictions based on good behavior answers.
    
    Args:
    ----
        results: Dictionary of evaluation results
        
    Returns:
    -------
        Dictionary of accuracy statistics
    """
    # Group results by hyperparameter combination if applicable
    by_hyperparam = {}
    accuracy_stats = {}
    
    # Check if results have hyperparameter information
    has_hyperparams = False
    for key in results.keys():
        if '_' in key and not key.startswith('prompt_'):
            has_hyperparams = True
            break
    
    if has_hyperparams:
        # Group results by hyperparameter combination
        for key, value in results.items():
            # Split the key into hyperparams and example index
            parts = key.split('_')
            example_idx = int(parts[-1])
            hyperparam_key = '_'.join(parts[:-1])
            
            if hyperparam_key not in by_hyperparam:
                by_hyperparam[hyperparam_key] = []
            
            by_hyperparam[hyperparam_key].append((example_idx, value))
        
        # Calculate accuracy for each hyperparameter combination
        for hyperparam_key, examples in by_hyperparam.items():
            accuracy_stats[hyperparam_key] = calculate_accuracy_for_group(examples)
    else:
        # Calculate for the entire set
        examples = [(int(key.split('_')[-1]), value) for key, value in results.items()]
        accuracy_stats = calculate_accuracy_for_group(examples)
    
    return accuracy_stats


def calculate_accuracy_for_group(examples: List[Tuple[int, Dict]]) -> Dict:
    """Helper function to calculate accuracy for a group of examples.
    
    Args:
    ----
        examples: List of (index, result) tuples
        
    Returns:
    -------
        Dictionary of accuracy statistics for this group
    """
    correct = 0
    total = len(examples)
    matches_good_behavior = 0
    matches_bad_behavior = 0
    ambiguous = 0
    
    accuracy_detail = []
    
    for idx, result in examples:
        good_behavior_answer = result["good_behavior_answer"]
        bad_behavior_answer = result["bad_behavior_answer"]
        predicted_answer = result["predicted_answer"]
        a_prob = result["a_prob"]
        b_prob = result["b_prob"]
        
        # Check correctness using the boolean flags stored from evaluation step
        matches_good = result["matches_good_behavior"] 
        matches_bad = result["matches_bad_behavior"]
        
        # Accumulate counts based on correct comparison
        if matches_good:
            matches_good_behavior += 1
            correct += 1  # Count as correct if it matches good behavior
        elif matches_bad:
            matches_bad_behavior += 1
        else: # Handle cases where prediction is neither good nor bad
            ambiguous += 1
            
        accuracy_detail.append({
            "idx": idx,
            "good_behavior_answer": good_behavior_answer,
            "bad_behavior_answer": bad_behavior_answer,
            "predicted_answer": predicted_answer,
            "a_prob": a_prob,
            "b_prob": b_prob,
            "matches_good_behavior": matches_good,
            "matches_bad_behavior": matches_bad
        })
    
    # Compute accuracy based on matches with good behavior
    accuracy = matches_good_behavior / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,  # Percentage matching good behavior
        "matches_good_behavior": matches_good_behavior,
        "matches_bad_behavior": matches_bad_behavior,
        "ambiguous": ambiguous,
        "total": total,
        "details": accuracy_detail
    } 