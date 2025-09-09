"""Activation collection and steering function creation utilities."""

from typing import Dict, List, Any
import torch

from hookedllm.config import ModuleSelection, TokenSelection, CachingConfig
from hookedllm.config import MeanActivationConfig, ConceptorConfig
from hookedllm.steering_functions import (
    MeanActivationSteering, 
    ConceptorSteering, 
    SteeringFunction, 
    merge_conceptors_and_not, 
    compute_conceptor
)


def collect_activations(model, examples: List[str], caching_config: CachingConfig, max_examples: int = None) -> Dict[str, torch.Tensor]:
    """Collect activations for a list of examples.
    
    Args:
    ----
        model: The model to collect activations from
        examples: List of input texts
        caching_config: Configuration for caching
        max_examples: Maximum number of examples to process
        
    Returns:
    -------
        Dictionary of activations keyed by module name
    """
    if max_examples is not None:
        examples = examples[:max_examples]
    
    print(f"Collecting activations for {len(examples)} examples...")
    with model.caching(caching_config):
        # TODO: process examples in batches for efficiency
        for example in examples:
            model.forward(example)
    
    # Get and format the cached activations
    activations = model.get_and_clear_cache()
    activations = {
        k: torch.cat(v, dim=0)
        for k, v in activations.items()
    }
    
    return activations


def create_activation_steering_function(
    config: MeanActivationConfig,
    positive_activations: Dict[str, torch.Tensor],
    negative_activations: Dict[str, torch.Tensor]
) -> MeanActivationSteering:
    """
    Create an activation (vector) steering function from positive and negative examples.
    For activation steering, we compute one direction from the difference of means.
    
    Args:
    ----
        config: Configuration for mean activation steering
        positive_activations: Activations for positive examples (good behavior)
        negative_activations: Activations for negative examples (bad behavior)
        
    Returns:
    -------
        MeanActivationSteering object initialized with the mean direction
    """
    # Compute mean activations for each class
    mean_positive = {k: v.mean(dim=0, keepdim=True) for k, v in positive_activations.items()}
    mean_negative = {k: v.mean(dim=0, keepdim=True) for k, v in negative_activations.items()}
    
    # Compute difference of means (positive - negative)
    diff_activations = {
        k: mean_positive[k] - mean_negative[k] 
        for k in positive_activations.keys()
    }
    
    # Create steering function using the direction (positive - negative)
    return MeanActivationSteering(
        config=config,
        activations=diff_activations,
    )


def create_conceptor_steering_function(
    config: ConceptorConfig,
    positive_activations: Dict[str, torch.Tensor],
    negative_activations: Dict[str, torch.Tensor],
    use_pos_neg_conceptors: bool = False
) -> ConceptorSteering:
    """
    Create a conceptor steering function from positive and negative examples.
    
    If use_pos_neg_conceptors is True, computes C_pos AND (NOT C_neg).
    Otherwise, computes conceptor based on positive_activations only.
    
    Args:
    ----
        config: Configuration for the conceptor
        positive_activations: Activations for positive examples (good behavior)
        negative_activations: Activations for negative examples (bad behavior)
        use_pos_neg_conceptors: Whether to use positive AND (NOT negative) conceptors

    Returns:
    -------
        ConceptorSteering object initialized with the appropriate conceptors
    """
    target_conceptors = {}

    if use_pos_neg_conceptors:
        print("Computing conceptors using C_pos AND (NOT C_neg) strategy.")
        # Compute positive and negative conceptors for each module
        pos_conceptors = {
            module_name: compute_conceptor(acts, config.aperture)
            for module_name, acts in positive_activations.items()
        }
        neg_conceptors = {
            module_name: compute_conceptor(acts, config.aperture)
            for module_name, acts in negative_activations.items()
        }

        # Merge them using AND NOT for each module
        merged_conceptors = {}
        for module_name in pos_conceptors:
            if module_name not in neg_conceptors:
                print(f"Warning: Module {module_name} found in positive but not negative activations. Skipping.")
                continue
            merged_conceptors[module_name] = merge_conceptors_and_not(
                pos_conceptors[module_name], neg_conceptors[module_name]
            )
        target_conceptors = merged_conceptors

    else:
        print("Computing conceptors using the difference of mean activations (positive - negative).")
        # Compute mean activations for each class
        mean_positive = {k: v.mean(dim=0, keepdim=True) for k, v in positive_activations.items()}
        mean_negative = {k: v.mean(dim=0, keepdim=True) for k, v in negative_activations.items()}
        
        # Compute difference of means (positive - negative)
        diff_activations = {
            k: mean_positive[k] - mean_negative[k] 
            for k in positive_activations.keys()
        }
        
        # Compute conceptors on the difference of means
        target_conceptors = {
            module_name: compute_conceptor(acts, config.aperture)
            for module_name, acts in diff_activations.items()
        }

    # Initialize ConceptorSteering with the computed conceptors
    return ConceptorSteering(
        config=config,
        conceptors=target_conceptors,
    )


def create_steering_function(
    steering_function_type: str,
    module_selection: ModuleSelection,
    positive_activations: Dict[str, torch.Tensor],
    negative_activations: Dict[str, torch.Tensor],
    hyperparam_dict: Dict[str, Any],
    use_additive_conceptor: bool = False,
    use_pos_neg_conceptors: bool = False
) -> SteeringFunction:
    """Create the appropriate steering function based on type.
    
    Args:
    ----
        steering_function_type: Type of steering function ("activation" or "conceptor")
        module_selection: Configuration for module selection
        positive_activations: Activations for positive examples (good behavior)
        negative_activations: Activations for negative examples (bad behavior)
        hyperparam_dict: Dictionary of hyperparameters (beta, aperture)
        use_additive_conceptor: Whether to use additive conceptor steering
        use_pos_neg_conceptors: Whether to compute C_pos AND (NOT C_neg) for conceptor steering
        
    Returns:
    -------
        Appropriate steering function initialized with parameters
    """
    if steering_function_type == "activation":
        config = MeanActivationConfig(
            modules=module_selection,
            tokens=TokenSelection(position=[-1]),
            **hyperparam_dict,
        )
        return create_activation_steering_function(
            config=config,
            positive_activations=positive_activations, 
            negative_activations=negative_activations
        )
    
    elif steering_function_type == "conceptor":
        config = ConceptorConfig(
            modules=module_selection,
            tokens=TokenSelection(),
            additive=use_additive_conceptor,
            **hyperparam_dict,
        )
        return create_conceptor_steering_function(
            config=config,
            positive_activations=positive_activations,
            negative_activations=negative_activations,
            use_pos_neg_conceptors=use_pos_neg_conceptors
        )
    
    else:
        raise ValueError(f"Invalid steering function type: {steering_function_type}") 