"""Utility functions for HookedLLM."""

# Import all functions from module files
from .module_selection import (
    validate_module_selection,
    get_modules,
    get_token_positions
)

from .dataset import (
    get_examples,
    get_test_prompts,
    get_open_ended_test_prompts
)

from .activations import (
    collect_activations,
    create_activation_steering_function,
    create_conceptor_steering_function,
    create_steering_function
)

from .serialization import (
    save_steering_function,
    load_steering_function
)

from .evaluation import (
    evaluate_prompts_with_logits,
    calculate_accuracy,
    calculate_accuracy_for_group
)

# Constants
BOLD = '\033[1m'
RESET = '\033[0m'
BLUE = '\033[94m'

# Export all imported functions
__all__ = [
    # module_selection.py
    'validate_module_selection',
    'get_modules',
    'get_token_positions',
    
    # dataset.py
    'get_examples',
    'get_test_prompts',
    'get_open_ended_test_prompts',
    
    # activations.py
    'collect_activations',
    'create_activation_steering_function',
    'create_conceptor_steering_function',
    'create_steering_function',
    
    # serialization.py
    'save_steering_function',
    'load_steering_function',
    
    # evaluation.py
    'evaluate_prompts_with_logits',
    'calculate_accuracy',
    'calculate_accuracy_for_group',
    
    # constants
    'BOLD',
    'RESET',
    'BLUE',
] 