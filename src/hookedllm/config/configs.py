"""Configuration classes for steering."""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union


@dataclass
class TokenSelection:
    """Configuration for token selection/targeting.

    NOTE: position is counted relative to either input or output tokens.

    Examples
    --------
    tokens = TokenSelection(position=-1) # Last token
    tokens = TokenSelection(position=[0, 1, 2]) # First 3 tokens
    tokens = TokenSelection(position=[-3, -2, -1]) # Last 3 tokens
    tokens = TokenSelection(token=":") # All occurrences of ":"
    tokens = TokenSelection(token=":", occurrence=1) # First occurrence of ":"
    tokens = TokenSelection(token=":", occurrence=-1) # Last occurrence of ":"

    """

    position: Optional[Union[List[int], int, str, Callable]] = None  # Default to last token
    token: Optional[str] = None  # Target token string (e.g., ":")
    occurrence: Optional[Union[int, List[int]]] = None  # Which occurrence to target (-1 for last)

    def __post_init__(self):
        """Post-initialization validation."""
        if self.position is not None and self.token is not None:
            raise ValueError("Only one of position or token can be provided")
        elif isinstance(self.occurrence, list):
            raise NotImplementedError(
                "Occurrence targeting is not implemented for list of occurrences"
            )


@dataclass
class ModuleSelection:
    """Configuration for hooks.
     f"{module_selection.layer_path}.{layer_idx}.{layer_module}" adjust accordingly
    """

    layers: List[int] = field(default_factory=list)
    layer_modules: List[str] = field(default_factory=list)
    global_modules: List[str] = field(default_factory=list)
    layer_path: str = "model.layers"


@dataclass
class CachingConfig:
    """Configuration for caching."""

    modules: ModuleSelection = field(default_factory=ModuleSelection)
    tokens: TokenSelection = field(default_factory=TokenSelection)

    # # Caching parameters - TODO: future work
    # cache_dir: Optional[str] = None
    # cache_threshold: int = 1000  # Number of samples before disk offloading


@dataclass
class SteeringConfig:
    """Base configuration for all steering mechanisms."""

    modules: ModuleSelection = field(default_factory=ModuleSelection)
    tokens: TokenSelection = field(default_factory=TokenSelection)

    # General steering parameters
    beta: Optional[float] = None  # Steering strength/intensity
    mean_center: bool = False  # Whether to apply mean-centering

    def __post_init__(self):
        """Post-initialization validation."""
        if self.tokens.position is not None and self.tokens.position != 0:
            raise NotImplementedError("Steering at specific positions is not implemented yet")
        if self.tokens.token is not None or self.tokens.occurrence is not None:
            raise NotImplementedError("Token/occurrence targeting is not implemented yet")


@dataclass
class MeanActivationConfig(SteeringConfig):
    """Configuration specific to mean activation steering."""

    steering_type: str = "mean_activation"


@dataclass
class ConceptorConfig(SteeringConfig):
    """Configuration specific to conceptor-based steering."""

    steering_type: str = "conceptor"
    aperture: float = 10.0  # Regularization parameter for conceptor matrix
    additive: bool = False  # if True, x+C@x is used instead of C@x

    # # Boolean operation parameters - TODO: future work
    # combine_method: Optional[str] = None  # "and", "or", "not"
    # combine_weights: Optional[Dict[str, float]] = None
