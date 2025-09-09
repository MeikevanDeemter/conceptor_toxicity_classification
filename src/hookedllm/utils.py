"""Utility functions for HookedLLM (imports from submodules).

NOTE: this file is not used in the library, but is used for backward compatibility.
Once we add tests using the new utils folder, we can remove this file.
"""

# Re-export all functions from utils submodules for backward compatibility
from hookedllm.utils import (
    # module_selection.py
    validate_module_selection,
    get_modules,
    get_token_positions,
    
    # dataset.py
    get_examples,
    get_test_prompts,
    get_open_ended_test_prompts,
    
    # activations.py
    collect_activations,
    create_activation_steering_function,
    create_conceptor_steering_function,
    create_steering_function,
    
    # serialization.py
    save_steering_function,
    load_steering_function,
    
    # evaluation.py
    evaluate_prompts_with_logits,
    calculate_accuracy,
    calculate_accuracy_for_group,
    
    # constants
    BOLD,
    RESET,
    BLUE,
)
