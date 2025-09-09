"""Tests for save and load functionality of steering functions."""

import os
import tempfile
from typing import Callable, Dict, Tuple, Type, Union

import pytest
import torch
import numpy as np

from hookedllm.config import ConceptorConfig, MeanActivationConfig, ModuleSelection, TokenSelection
from hookedllm.steering_functions import ConceptorSteering, MeanActivationSteering, SteeringFunction


def create_dummy_steering_fn(
    steering_class: Type[SteeringFunction],
    dim: int = 768, 
    num_modules: int = 2,
    seed: int = 42
) -> SteeringFunction:
    """Create a dummy steering function with random activations.
    
    Args:
        steering_class: The class of steering function to create
        dim: The dimension of the activations
        num_modules: The number of modules to create
        seed: Random seed for reproducibility
        
    Returns:
        A steering function of the specified class
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create dummy module selection
    module_sel = ModuleSelection(
        layers=[i for i in range(num_modules)],
        layer_modules=["attention"],
        layer_path="model.layers"
    )
    
    # Create token selection
    token_sel = TokenSelection()
    
    # Create config based on steering function class
    if steering_class == ConceptorSteering:
        config = ConceptorConfig(
            modules=module_sel,
            tokens=token_sel,
            aperture=10.0,
            beta=1.0,
            additive=False
        )
    elif steering_class == MeanActivationSteering:
        config = MeanActivationConfig(
            modules=module_sel,
            tokens=token_sel,
            beta=1.0
        )
    else:
        raise ValueError(f"Unsupported steering function class: {steering_class}")
    
    # Create dummy activations
    dummy_activations = {}
    for i in range(num_modules):
        module_name = f"model.layers.{i}.attention"
        # Create random activations (10 samples, each of dimension dim)
        dummy_activations[module_name] = torch.randn(10, dim)
    
    # Create steering function
    return steering_class(config=config, activations=dummy_activations)


def assert_steering_functions_equal(
    fn1: SteeringFunction, 
    fn2: SteeringFunction
):
    """Assert that two steering functions are equal.
    
    Args:
        fn1: First steering function
        fn2: Second steering function
    """
    # Check types
    assert type(fn1) == type(fn2), "Steering functions must be of the same type"
    
    # Test shared config properties
    assert fn1.config.beta == fn2.config.beta
    
    # Get the appropriate attribute name and additional config checks based on type
    if isinstance(fn1, ConceptorSteering):
        attr_name = "conceptors"
        # Test ConceptorSteering specific config
        assert fn1.config.aperture == fn2.config.aperture
        assert fn1.config.additive == fn2.config.additive
    elif isinstance(fn1, MeanActivationSteering):
        attr_name = "vectors"
    else:
        raise ValueError(f"Unsupported steering function type: {type(fn1)}")
    
    # Get the attributes
    attrs1 = getattr(fn1, attr_name)
    attrs2 = getattr(fn2, attr_name)
    
    # Check if they have the same keys
    assert set(attrs1.keys()) == set(attrs2.keys())
    
    # Check if the values are equal
    for key in attrs1:
        assert torch.allclose(attrs1[key], attrs2[key])
    
    # Test apply function on random inputs
    for key in attrs1:
        # Get the shape for test input
        shape = attrs1[key].shape[0]
        
        # Create random test input with fixed seed for reproducibility
        torch.manual_seed(42)
        test_input = torch.randn(3, shape)
        
        # Apply both functions
        output1 = fn1.apply(test_input, key)
        output2 = fn2.apply(test_input, key)
        
        # Check if outputs are equal
        assert torch.allclose(output1, output2)


@pytest.mark.parametrize(
    "steering_class,save_name", 
    [
        (ConceptorSteering, "test_conceptor"),
        (MeanActivationSteering, "test_mean_activation")
    ]
)
@pytest.mark.parametrize(
    "dim,num_modules", 
    [
        (768, 2),  # Default size
        (1024, 1), # Different dimension, one module
        (512, 3),  # Smaller dimension, more modules
    ]
)
def test_save_load_steering(steering_class, save_name, dim, num_modules):
    """Test saving and loading steering functions."""
    # Create a temporary directory for file operations
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create path for saving the steering function
        save_path = os.path.join(temp_dir, save_name)
        
        # Create a dummy steering function
        steering_fn = create_dummy_steering_fn(
            steering_class=steering_class,
            dim=dim, 
            num_modules=num_modules
        )
        
        # Save the steering function
        steering_fn.save(save_path)
        
        # Load the steering function using the built-in loader
        loaded_fn = steering_class.load(save_path)
        
        # Test that they're equal
        assert_steering_functions_equal(steering_fn, loaded_fn) 