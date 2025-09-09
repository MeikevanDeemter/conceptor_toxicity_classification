"""Module selection and token manipulation utilities."""

from typing import Dict, List
import torch

from hookedllm.config import ModuleSelection, TokenSelection


def validate_module_selection(model, module_selection: ModuleSelection):
    """Validate module selection config.

    Args:
    ----
        model: The model to validate the module selection for.
        module_selection: The module selection config to validate.

    Raises:
    ------
        ValueError: If the module selection config is invalid.

    Returns:
    -------
        True if the module selection config is valid, False otherwise.

    """
    module_keys = list(dict(model.named_modules()).keys())


    # Validate layer path
    if module_selection.layers and module_selection.layer_modules:
        if module_selection.layer_path not in module_keys:
            raise ValueError(f"layer_path {module_selection.layer_path} not found in model")

    # Validate layer modules
    for layer_idx in module_selection.layers:
        for layer_module in module_selection.layer_modules:
            module_path = f"{module_selection.layer_path}.{layer_idx}.{layer_module}".strip(".")
            if module_path not in module_keys:
                raise ValueError(f"layer module {layer_idx}.{layer_module} not found in model")

    # Validate global modules
    for global_module in module_selection.global_modules:
        if global_module not in module_keys:
            raise ValueError(f"global module {global_module} not found in model")

    return True


def get_modules(model, module_selection: ModuleSelection) -> Dict[str, torch.nn.Module]:
    """Get module objects based on module selection config.

    Args:
    ----
        model: The model to get the modules from.
        module_selection: The module selection config to get the modules from.

    Returns:
    -------
        A dictionary mapping module names to module objects.

    """
    validate_module_selection(model, module_selection)
    modules = {}
    all_modules = dict(model.named_modules())

    # TODO: implement this
    # # Handle special selections
    # if module_selection.layers in ("all", "*"):
    # if module_selection.global_modules in ("all", "*"):
    # if module_selection.layer_modules in ("all", "*"):

    # Handle layer-specific modules
    for layer_idx in module_selection.layers:
        for layer_module in module_selection.layer_modules:
            module_path = f"{module_selection.layer_path}.{layer_idx}.{layer_module}".strip(".")
            modules[module_path] = all_modules[module_path]

    # Handle global modules
    for global_module in module_selection.global_modules:
        modules[global_module] = all_modules[global_module]

    return modules


def get_token_positions(
    tokenizer,
    input_ids: torch.Tensor,
    token_selection: TokenSelection,
    default_selection: str = "all",
) -> List[List[int]]:
    """Find token positions based on token selection config.

    Args:
    ----
        tokenizer: The tokenizer to use.
        input_ids: The input token IDs to find positions in, of shape (batch_size, seq_len).
        token_selection: The token selection config.
        default_selection: The default selection to use if no other selection criteria are met.

    Returns:
    -------
        A list of token positions for each batch element.

    """
    assert default_selection in ["last", "all"], "default_selection must be 'last' or 'all'"

    if token_selection.position is not None:
        # Handle position-based selection
        if isinstance(token_selection.position, int):
            # Single position (e.g., -1 for last token)
            if token_selection.position > input_ids.shape[1]:
                raise ValueError(f"position {token_selection.position} >> {input_ids.shape[1]}")
            elif token_selection.position < -input_ids.shape[1]:
                raise ValueError(f"position {token_selection.position} << {-input_ids.shape[1]}")
            elif token_selection.position < 0:
                pos = input_ids.shape[1] + token_selection.position
            else:
                pos = token_selection.position
            return [pos] * input_ids.shape[0]  # Same position for all batch elements
        elif isinstance(token_selection.position, list):
            # Multiple positions
            positions = []
            for pos in token_selection.position:
                if pos > input_ids.shape[1]:
                    raise ValueError(f"position {pos} >> {input_ids.shape[1]}")
                elif pos < -input_ids.shape[1]:
                    raise ValueError(f"position {pos} << {-input_ids.shape[1]}")
                elif pos < 0:
                    pos = input_ids.shape[1] + pos
                positions.append(pos)
            return [positions] * input_ids.shape[0]
        else:
            raise ValueError("Invalid type for position, expected `int` or `List[int]`.")

    elif token_selection.token is not None:
        # Handle token-based selection (find all occurrences of a specific token)
        token_id = tokenizer.convert_tokens_to_ids(token_selection.token)
        positions = []
        for i in range(input_ids.shape[0]):
            # Find where input_ids matches token_id
            matches = (input_ids[i] == token_id).nonzero(as_tuple=True)[0]

            if len(matches) == 0:
                # No matches, use last token as fallback
                positions.append(input_ids.shape[1] - 1)
            elif token_selection.occurrence is not None:
                # Specific occurrence
                idx = token_selection.occurrence
                if idx < 0:  # Negative indexing
                    idx = len(matches) + idx
                if 0 <= idx < len(matches):
                    positions.append(matches[idx].item())
                else:
                    positions.append(input_ids.shape[1] - 1)  # Fallback to last token
            else:
                # All occurrences (return first for now)
                positions.append(matches[0].item())

        return positions

    # Default to last token if no selection criteria
    if default_selection == "last":
        return [input_ids.shape[1] - 1] * input_ids.shape[0]
    elif default_selection == "all":
        return [list(range(input_ids.shape[1]))] * input_ids.shape[0] 