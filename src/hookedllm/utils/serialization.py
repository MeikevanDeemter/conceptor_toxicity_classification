"""Serialization utilities for saving and loading steering functions."""

from typing import Dict, Any
from pathlib import Path
import torch

from hookedllm.config import ModuleSelection, TokenSelection
from hookedllm.config import MeanActivationConfig, ConceptorConfig
from hookedllm.steering_functions import ConceptorSteering, MeanActivationSteering, SteeringFunction


def save_steering_function(steering_fn: SteeringFunction, save_dir: Path, config_info: Dict[str, Any]) -> str:
    """Save a steering function and its configuration to disk.
    
    Args:
    ----
        steering_fn: Steering function to save
        save_dir: Directory to save to
        config_info: Dictionary of configuration information
        
    Returns:
    -------
        Parameter string used in the filename
    """
    # Create a unique identifier for this steering function
    params_str = "_".join([f"{k}={v}" for k, v in config_info.items() if k != "args"])
    
    # Extract the configuration and conceptors
    if isinstance(steering_fn, ConceptorSteering):
        state = {
            "config": {
                "modules": steering_fn.config.modules.layers,
                "layer_modules": steering_fn.config.modules.layer_modules,
                "layer_path": steering_fn.config.modules.layer_path,
                "aperture": steering_fn.config.aperture,
                "beta": steering_fn.config.beta,
                "additive": steering_fn.config.additive,
            },
            "conceptors": {
                name: C.detach().cpu() 
                for name, C in steering_fn.conceptors.items()
            }
        }
    elif isinstance(steering_fn, MeanActivationSteering):
        state = {
            "config": {
                "modules": steering_fn.config.modules.layers,
                "layer_modules": steering_fn.config.modules.layer_modules,
                "layer_path": steering_fn.config.modules.layer_path,
                "beta": steering_fn.config.beta,
            },
            "activations": {
                name: act.detach().cpu()
                for name, act in steering_fn.activations.items()
            }
        }
    else:
        raise ValueError(f"Unsupported steering function type: {type(steering_fn)}")
    
    torch.save(state, save_dir / f"steering_fn_{params_str}.pt")
    return params_str


def load_steering_function(path: Path, device: str = "cpu") -> SteeringFunction:
    """Load a steering function from a file.
    
    Args:
    ----
        path: Path to the saved steering function
        device: Device to load the steering function to
        
    Returns:
    -------
        Loaded steering function
    """
    state = torch.load(path, map_location=device)
    cfg = state["config"]

    module_sel = ModuleSelection(
        layers=cfg["modules"],
        layer_modules=cfg["layer_modules"],
        layer_path=cfg["layer_path"],
    )

    token_sel = TokenSelection()

    # Determine the type of steering function based on saved state
    if "conceptors" in state:
        concep_cfg = ConceptorConfig(
            modules=module_sel,
            tokens=token_sel,
            aperture=cfg["aperture"],
            beta=cfg.get("beta", 1.0),
            additive=cfg["additive"],
        )

        steering_fn = ConceptorSteering(
            config=concep_cfg,
            conceptors={ 
                name: C.to(device) 
                for name, C in state["conceptors"].items() 
            }
        )
    elif "activations" in state:
        act_cfg = MeanActivationConfig(
            modules=module_sel,
            tokens=token_sel,
            beta=cfg.get("beta", 1.0),
        )

        steering_fn = MeanActivationSteering(
            config=act_cfg,
            activations={
                name: act.to(device)
                for name, act in state["activations"].items()
            }
        )
    else:
        raise ValueError(f"Unknown steering function type in file: {path}")
    
    return steering_fn 