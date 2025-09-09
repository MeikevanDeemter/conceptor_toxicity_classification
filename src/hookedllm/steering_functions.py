"""Base class for steering functions."""

import json
import os
from dataclasses import asdict, is_dataclass
from typing import Dict, Optional, Tuple, Any, TypeVar, Type, cast

import torch

from hookedllm.config import ConceptorConfig, MeanActivationConfig, SteeringConfig
from hookedllm.config import ModuleSelection, TokenSelection


T = TypeVar('T')


def dataclass_from_dict(dataclass_type: Type[T], data: Dict[str, Any]) -> T:
    """Recursively instantiate nested dataclasses from dictionaries.
    
    Args:
        dataclass_type: The dataclass type to instantiate
        data: Dictionary containing the data
        
    Returns:
        An instance of the dataclass
    """
    if not data:
        return cast(T, None)
    
    # Get field types from the dataclass
    field_types = {field.name: field.type for field in dataclass_type.__dataclass_fields__.values()}
    
    # Process each field in the input dictionary
    kwargs = {}
    for key, value in data.items():
        # Skip unknown fields
        if key not in field_types:
            continue
            
        field_type = field_types[key]
        
        # If the value is a dictionary and the field type is a dataclass, recursively process it
        if isinstance(value, dict) and is_dataclass(field_type):
            kwargs[key] = dataclass_from_dict(field_type, value)
        else:
            kwargs[key] = value
            
    return dataclass_type(**kwargs)


class SteeringFunction:
    """Base class for steering functions/matrices.

    This class provides a common interface for all steering functions.
    Subclasses implement specific steering behaviors.

    Attributes
    ----------
        config: SteeringConfig for configuration

    """

    def __init__(self, config: SteeringConfig, **kwargs):
        """Initialize a steering function.

        Args:
        ----
            config: SteeringConfig for configuration
            **kwargs: Keyword arguments for the steering function

        """
        self.config = config

    def apply(self, x: torch.Tensor, module_name: str) -> torch.Tensor:
        """Apply steering to activation x.

        Args:
        ----
            x: Activation tensor to modify
            module_name: Name of the module

        Returns:
        -------
            Modified activation tensor

        """
        raise NotImplementedError

    def _save_objects(self, objects: Dict[str, torch.Tensor], path: str):
        """Save steering function to disk.

        Args:
        ----
            objects: objects to save for this steering function
            path: Path to save the steering function

        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        try:
            # Save vectors using safetensors if available
            from safetensors.torch import save_file

            save_file(objects, f"{path}.safetensors")
        except ImportError:
            # Fall back to PyTorch saving
            torch.save(objects, f"{path}.pt")

        # Save config separately
        with open(f"{path}_config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=4)

    def save(self, path: str):
        """Save steering function to disk.

        Calls _save_objects with objects to saved for a given steering function.

        Args:
        ----
            path: Path to save the steering function

        """
        raise NotImplementedError

    @classmethod
    def _load_objects(cls, path: str):
        """Load steering function config and objects from disk.

        Args:
        ----
            path: Path to load the steering function from

        Returns:
        -------
            config: SteeringConfig
            objects: Dictionary mapping module names to steering objects

        """
        # Load config
        with open(f"{path}_config.json", "r") as f:
            config_dict = json.load(f)

        # Determine the config type based on the steering_type
        if config_dict.get("steering_type") == "mean_activation":
            config = dataclass_from_dict(MeanActivationConfig, config_dict)
        elif config_dict.get("steering_type") == "conceptor":
            config = dataclass_from_dict(ConceptorConfig, config_dict)
        else:
            raise ValueError(f"Unknown steering type: {config_dict.get('steering_type')}")

        # Load objects
        try:
            from safetensors.torch import load_file
            objects = load_file(f"{path}.safetensors")
        except (ImportError, FileNotFoundError):
            # Fall back to PyTorch loading
            objects = torch.load(f"{path}.pt")

        return config, objects

    @classmethod
    def load(cls, path: str):
        """Load steering function from disk.

        Calls _load_objects with path to load the steering function from.

        Args:
        ----
            path: Path to load the steering function from

        Returns:
        -------
            A new instance of the steering function

        """
        raise NotImplementedError


def compute_conceptor(activations, aperture):
    """Compute the conceptor matrix for a given input matrix.

    Calculation done in fp32 on CPU, then converted back to the correct dtype and device.

    Args:
    ----
        activations (torch.Tensor): Input matrix of shape (N, D).
        aperture (float): Aperture parameter for the conceptor.

    Returns:
    -------
        torch.Tensor: Conceptor matrix of shape (D, D).

    """
    X = activations.cpu().to(dtype=torch.float32)
    R = torch.matmul(X.T, X) / X.shape[0]
    U, S, _ = torch.svd(R)
    C = U * (S / (S + (aperture ** (-2)) * torch.ones(S.shape, device=X.device))) @ U.T
    return C.to(device=activations.device, dtype=activations.dtype)


def merge_conceptors_and(c1_matrix: torch.Tensor, c2_matrix: torch.Tensor) -> torch.Tensor:
    """
    Merges two conceptor matrices using the logical AND operation.

    Equation: C1 AND C2 = pinv(pinv(C1) + pinv(C2) + I)
    Uses pseudo-inverse (pinv) for numerical stability.

    Args:
        c1_matrix: The first conceptor matrix (torch.Tensor).
        c2_matrix: The second conceptor matrix (torch.Tensor).

    Returns:
        The merged conceptor matrix (torch.Tensor).

    Raises:
        ValueError: If matrices are not square or have incompatible shapes.
        NotImplementedError: This function is not yet implemented.
    """
    # 1. Check if matrices have compatible shapes (square and same size).
    if (
        len(c1_matrix.shape) != 2
        or c1_matrix.shape[0] != c1_matrix.shape[1]
        or c1_matrix.shape != c2_matrix.shape
    ):
        raise ValueError(
            "Conceptor matrices must be square and have the same shape. "
            f"Got shapes {c1_matrix.shape} and {c2_matrix.shape}."
        )

    # --- Implementation Steps --- 
    # 2. Get the dimension n.
    n = c1_matrix.shape[0]
    
    # 3. Get the identity matrix I. Ensure same dtype and device.
    identity = torch.eye(n, dtype=c1_matrix.dtype, device=c1_matrix.device)
    
    # 4. Calculate the pseudo-inverses C1^-1 and C2^-1.
    # Use fp32 for potentially better numerical stability during inversion
    c1_pinv = torch.linalg.pinv(c1_matrix.to(torch.float32))
    c2_pinv = torch.linalg.pinv(c2_matrix.to(torch.float32))
    
    # 5. Compute the sum S = C1^-1 + C2^-1 + I.
    sum_matrix = c1_pinv + c2_pinv + identity.to(torch.float32)
    
    # 6. Calculate the pseudo-inverse of the sum: S^-1.
    merged_matrix = torch.linalg.pinv(sum_matrix)
    
    # 7. Return the resulting merged matrix, converting back to original dtype.
    return merged_matrix.to(c1_matrix.dtype)


def negate_conceptor(c_matrix: torch.Tensor) -> torch.Tensor:
    """
    Negates a conceptor matrix using the logical NOT operation.

    Equation: NOT C = I - C

    Args:
        c_matrix: The conceptor matrix to negate (torch.Tensor).

    Returns:
        The negated conceptor matrix (torch.Tensor).

    Raises:
        ValueError: If the matrix is not square.
    """
    # 1. Check if the matrix is square.
    if len(c_matrix.shape) != 2 or c_matrix.shape[0] != c_matrix.shape[1]:
        raise ValueError(
            f"Conceptor matrix must be square. Got shape {c_matrix.shape}."
        )

    # 2. Get the dimension n.
    n = c_matrix.shape[0]

    # 3. Get the identity matrix I.
    identity = torch.eye(n, dtype=c_matrix.dtype, device=c_matrix.device)

    # 4. Calculate and return I - C.
    return identity - c_matrix


def merge_conceptors_and_not(c1_matrix: torch.Tensor, c2_matrix: torch.Tensor) -> torch.Tensor:
    """
    Merges two conceptor matrices using the logical AND NOT operation (C1 AND (NOT C2)).

    This is equivalent to C1 \\ (C1 AND C2) in some conceptor literature.
    Uses previously defined `negate_conceptor` and `merge_conceptors_and`.

    Args:
        c1_matrix: The first conceptor matrix (torch.Tensor).
        c2_matrix: The second conceptor matrix (the one to be negated) (torch.Tensor).

    Returns:
        The resulting merged conceptor matrix (torch.Tensor).

    Raises:
        ValueError: If matrices are not square or have incompatible shapes (via called functions).
    """
    # 1. Negate the second conceptor.
    # Checks for shape and squareness are handled within negate_conceptor.
    not_c2_matrix = negate_conceptor(c2_matrix)

    # 2. Merge the first conceptor with the negated second conceptor using AND.
    # Checks are handled within merge_conceptors_and.
    result_matrix = merge_conceptors_and(c1_matrix, not_c2_matrix)

    # 3. Return the result.
    return result_matrix


class ConceptorSteering(SteeringFunction):
    """Apply steering using conceptor matrices.

    This class implements conceptor-based steering for language model activations.
    """

    def __init__(
        self,
        config: ConceptorConfig,
        activations: Optional[Dict[str, torch.Tensor]] = None,
        conceptors: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize a ConceptorSteering object from activations or directly from conceptors.

        Args:
        ----
            config: ConceptorConfig for configuration
            activations: Dictionary mapping module names to activation tensors
            conceptors: Dictionary mapping module names to conceptor matrices

        """
        super().__init__(config)

        # Checks
        assert (activations is not None) or (conceptors is not None)
        if self.config.mean_center:
            raise NotImplementedError("Mean-centering is not implemented for ConceptorSteering.")

        # Initialize conceptor objects
        if activations is not None:
            assert self.config.aperture is not None
            self.conceptors = {
                module_name: compute_conceptor(activations[module_name], self.config.aperture)
                for module_name in activations
            }
        else:
            assert isinstance(conceptors, dict)
            self.conceptors = conceptors

    def apply(self, x: torch.Tensor, module_name: str) -> torch.Tensor:
        """Apply conceptor steering to activation x.

        Args:
        ----
            x: Activation tensor to modify
            module_name: Name of the module

        Returns:
        -------
            Modified activation tensor

        """
        if module_name not in self.conceptors:
            print(f"WARNING: {module_name} not in conceptors. Returning input tensor.")
            return x

        C = self.conceptors[module_name]
        if C.device != x.device:
            C = C.to(x.device)

        # Apply beta
        C = self.config.beta * C

        if self.config.additive:
            # Add identity matrix to conceptor for additive conceptor steering
            # (C + I) @ x = C @ x + x
            C = C + torch.eye(C.shape[0], device=C.device)

        # Check if we're working with batched input
        if len(x.shape) == 3:
            return torch.einsum("ij,abi->abj", C, x)
        elif len(x.shape) == 2:
            return torch.einsum("ij,bi->bj", C, x)
        else:
            return torch.matmul(C, x)

    def save(self, path: str):
        """Save steering function to disk.

        Args:
        ----
            path: Path to save the steering function

        """
        self._save_objects(self.conceptors, path)

    @classmethod
    def load(cls, path: str):
        """Load steering function from disk.

        Args:
        ----
            path: Path to load the steering function from

        Returns:
        -------
            A new instance of the steering function

        """
        config, conceptors = cls._load_objects(path)
        return cls(config=config, conceptors=conceptors)


class MeanActivationSteering(SteeringFunction):
    """Apply steering using mean activation vectors.

    This class implements mean activation vector steering for language model activations.
    """

    def __init__(
        self,
        config: MeanActivationConfig,
        activations: Optional[Dict[str, torch.Tensor]] = None,
        vectors: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Initialize a MeanActivationSteering object from activations or directly from vectors.

        Args:
        ----
            config: MeanActivationConfig for configuration
            activations: Dictionary mapping module names to activation tensors
            vectors: Dictionary mapping module names to steering vectors

        """
        super().__init__(config)

        # Checks
        assert (activations is not None) or (vectors is not None)
        if self.config.mean_center:
            raise NotImplementedError(
                "Mean-centering is not implemented for MeanActivationSteering."
            )

        # Initialize vectors
        if activations is not None:
            self.vectors = {
                module_name: torch.mean(
                    activations[module_name], dim=tuple(range(activations[module_name].dim() - 1))
                )
                for module_name in activations
            }
        else:
            assert isinstance(vectors, dict)
            self.vectors = vectors

    def apply(self, x: torch.Tensor, module_name: str) -> torch.Tensor:
        """Apply mean activation steering to activation x.

        Args:
        ----
            x: Activation tensor to modify
            module_name: Name of the module

        Returns:
        -------
            Modified activation tensor

        """
        if module_name not in self.vectors:
            print(f"WARNING: {module_name} not in vectors. Returning input tensor.")
            return x

        v = self.vectors[module_name]
        if v.device != x.device:
            v = v.to(x.device)

        return self.config.beta * v + x

    def save(self, path: str):
        """Save steering function to disk.

        Args:
        ----
            path: Path to save the steering function

        """
        self._save_objects(self.vectors, path)

    @classmethod
    def load(cls, path: str):
        """Load steering function from disk.

        Args:
        ----
            path: Path to load the steering function from

        Returns:
        -------
            A new instance of the steering function

        """
        config, vectors = cls._load_objects(path)
        return cls(config=config, vectors=vectors)
