"""HookedLLM is a lightweight library for extracting and steering activations from LLMs."""

__version__ = "0.1.0"

# Import and export main classes
from .HookedLLM import HookedLLM

# Import and export config classes
from .config import (
    CachingConfig,
    ModuleSelection,
    SteeringConfig,
    TokenSelection
)

# Import and export steering functions
from .steering_functions import SteeringFunction

# Import utility functions
from .utils import (
    get_modules,
    get_token_positions,
    collect_activations,
    create_activation_steering_function,
    create_conceptor_steering_function,
    create_steering_function,
    save_steering_function,
    load_steering_function,
    evaluate_prompts_with_logits,
    calculate_accuracy,
    calculate_accuracy_for_group
)

__all__ = [
    # Main class
    'HookedLLM',
    
    # Config classes
    'CachingConfig',
    'ModuleSelection',
    'SteeringConfig',
    'TokenSelection',
    
    # Steering function classes
    'SteeringFunction',
    
    # Utility functions
    'get_modules',
    'get_token_positions',
    'collect_activations',
    'create_activation_steering_function',
    'create_conceptor_steering_function',
    'create_steering_function',
    'save_steering_function',
    'load_steering_function',
    'evaluate_prompts_with_logits',
    'calculate_accuracy',
    'calculate_accuracy_for_group'
] 