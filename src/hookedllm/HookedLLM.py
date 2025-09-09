"""HookedLLM is a lightweight library for extracting and steering activations from LLMs.

TODOs:
- replace print with logging
- for multi-position steering, batch for efficiency
- why does hooking into transformer.h.1 not work? input is empty tuple. ln_1 works.
- implement */all for module selection
"""

from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hookedllm.config import CachingConfig, ModuleSelection, SteeringConfig, TokenSelection
from hookedllm.steering_functions import SteeringFunction
from hookedllm.utils import get_modules, get_token_positions


class HookedLLM:
    """HookedLLM is a lightweight library for extracting and steering activations from LLMs."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        debug_mode: bool = False,
        model_kwargs: Dict[str, Any] = {},
        tokenizer_kwargs: Dict[str, Any] = {"padding_side": "left"},
    ):
        """Initialize HookedLLM.

        Args:
        ----
            model_id: The ID of the model to load.
            model: The model to use.
            tokenizer: The tokenizer to use.
            debug_mode: Whether to enable debug mode.
            model_kwargs: Keyword arguments for the model.
            tokenizer_kwargs: Keyword arguments for the tokenizer.

        """
        self.debug_mode = debug_mode
        # Core objects
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = defaultdict(list)
        self.cache = defaultdict(list)

        # State
        self.caching_config = None  # currently active caching config
        self.steering_config = None  # currently active steering config
        self.modules = {}  # currently hooked modules
        self.steering_function = None  # currently active steering function
        self.current_input_ids = None  # input ids of current input

        # Metadata
        self.model_id = model_id
        if self.model_id is None:
            self._debug("No model_id provided, using model.config.name_or_path")
            self.model_id = self.model.config.name_or_path

        # Initialize core objects
        if model is None:
            if "device_map" not in model_kwargs:
                # By default, move model to available accelerator
                model_kwargs["device_map"] = "auto"
            print(f"Loading model {model_id}...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs,
            )
        if tokenizer is None:
            print(f"Loading tokenizer {model_id}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                **tokenizer_kwargs,
            )
            # set pad token to eos token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            # check if tokenizer adds bos token
            if self.tokenizer.bos_token is None:
                self.tokenizer.bos_token = self.tokenizer.eos_token
            if self.tokenizer.bos_token_id is None:
                self.tokenizer.bos_token_id = self.tokenizer.eos_token_id

    def to_tokens(self, text: Union[str, list[str]], **kwargs):
        """Convert text to token IDs.

        Args:
        ----
            text: The text to convert to token IDs.
            **kwargs: Keyword arguments for the tokenizer.

        Returns:
        -------
            The token IDs.

        """
        # if tokenizer adds bos token when encoding, add it to the text
        # added a check incase the provided tokenizer does not have bos_token
        if len(self.tokenizer.encode("", **kwargs)) == 0 and self.tokenizer.bos_token is not None:
            if isinstance(text, str):
                text = self.tokenizer.bos_token + text
            elif isinstance(text, list):
                text = [self.tokenizer.bos_token + t for t in text]

        # NOTE: do we not want the attention mask, etc as well?
        if isinstance(text, str):
            return self.tokenizer.encode(
                text,
                return_tensors="pt",
                **kwargs,
            )
        elif isinstance(text, list):
            # NOTE: why batch_encode_plus?
            return self.tokenizer.batch_encode_plus(
                text,
                return_tensors="pt",
                padding=True,
                **kwargs,
            )["input_ids"]

    def forward(self, text: Union[str, list[str]], tokenizer_kwargs={}, model_kwargs={}):
        """Run a forward pass through the model.

        Args:
        ----
            text: The text to run through the model.
            tokenizer_kwargs: Keyword arguments for the tokenizer.
            model_kwargs: Keyword arguments for the model.

        Returns:
        -------
            The output of the model.

        """
        input_ids = self.to_tokens(text, **tokenizer_kwargs)
        input_ids = input_ids.to(self.model.device)
        self.current_input_ids = input_ids
        return self.model(input_ids, **model_kwargs)

    def generate(self, text: Union[str, list[str]], tokenizer_kwargs={}, generation_kwargs={}):
        """Generate text given a prompt.

        Args:
        ----
            text: The prompt to generate text from.
            tokenizer_kwargs: Keyword arguments for the tokenizer.
            generation_kwargs: Keyword arguments for the model.

        Returns:
        -------
            The generated text.

        """
        input_ids = self.to_tokens(text, **tokenizer_kwargs)
        input_ids = input_ids.to(self.model.device)
        self.current_input_ids = input_ids
        return self.model.generate(input_ids, **generation_kwargs)

    # Helper methods for hooks

    def _debug(self, message: str):
        """Print debug message if debug mode is enabled.

        Args:
        ----
            message: The message to print.

        """
        if self.debug_mode:
            print(f"[HookedLLM] {message}")

    # Activation caching hooks

    def _get_caching_hook(self, module_name: str):
        """Get the caching hook for a given module name.

        Args:
        ----
            module_name: The name of the module to be hooked.

        Returns:
        -------
            The caching hook for the given module name.

        """

        def _caching_hook(module, inputs):
            """Extract activations from module inputs.

            Reads:
                self.current_input_ids: The input IDs of the current input.
                self.modules: A dictionary of modules to be hooked.
                self.caching_config: The caching configuration.
                self.tokenizer: The tokenizer.

            Modifies:
                self.cache: The cache of activations.

            Args:
            ----
                module: The module that is being hooked.
                inputs: The inputs to the module.

            Returns:
            -------
                The inputs to the module.

            """
            if self.current_input_ids is None:
                raise ValueError("No input_ids available in caching hook (needed for position)")

            # Determine token positions to extract
            positions = get_token_positions(
                self.tokenizer, self.current_input_ids, self.caching_config.tokens
            )

            # Extract and store activations
            # Handle different output types (tuple vs tensor)
            inp = inputs[0] if isinstance(inputs, tuple) else inputs
            # Extract activation at position
            for idx, pos in enumerate(positions):
                activation = inp[idx, pos].detach().clone()
                self.cache[module_name].append(activation)
            self._debug(f"Cached activation for {module_name} at {positions=}")

            return inputs

        return _caching_hook

    def _setup_caching_hooks(self, caching_config: CachingConfig):
        """Set up hooks for activation extraction.

        Args:
        ----
            caching_config: The caching configuration.

        """
        self.caching_config = caching_config
        self.modules = get_modules(self.model, caching_config.modules)
        self._debug(f"Setting up caching hooks for {len(self.modules)} modules")

        # Clear any existing hooks
        self.clear_hooks()

        # Register hooks for each module
        for name, module in self.modules.items():
            self.hooks[name].append(module.register_forward_pre_hook(self._get_caching_hook(name)))
            self._debug(f"Registered caching hook for {name}")

    # Steering hooks

    def _get_steering_hook(self, module_name: str):
        """Get the steering hook for a given module name.

        Args:
        ----
            module_name: The name of the module to be hooked.

        Returns:
        -------
            The steering hook for the given module name.

        """

        def _steering_hook(module, inputs):
            """Apply steering to module inputs.

            Reads:
                self.steering_function: The steering function to apply.
                self.current_input_ids: The input IDs of the current input.
                self.steering_config: The steering configuration.
                self.tokenizer: The tokenizer.

            Modifies:
                self.model: The model.

            Args:
            ----
                module: The module that is being hooked.
                inputs: The inputs to the module.

            Returns:
            -------
                The inputs to the module.

            """
            if not self.steering_function or self.current_input_ids is None:
                raise ValueError("No steering functions or input_ids available in steering hook")

            # Handle different input types (tuple vs tensor)
            modified = isinstance(inputs, tuple)  # for correctly handling tuple outputs
            inp = inputs[0] if modified else inputs

            # Get token positions to apply steering
            if inp.shape[1] == 1:
                # Autoregressive generation, one token at a time
                if self.steering_config.tokens.position is None:
                    positions = [0]
                elif self.steering_config.tokens.position == 0:
                    positions = [0] if self.steering_count[module_name] == 0 else []
                else:
                    raise NotImplementedError("Steering at specific positions is not implemented")
            else:
                # Batch processing, multiple tokens at once
                positions = get_token_positions(
                    self.tokenizer, self.current_input_ids, self.steering_config.tokens
                )

            # Update steering count
            self.steering_count[module_name] += inp.shape[1]

            # Apply steering to those positions
            for idx, pos in enumerate(positions):
                x_steered = self.steering_function.apply(inp[idx, pos], module_name)
                inp[idx, pos] = x_steered
            self._debug(f"Applied steering for {module_name} at {positions=}")

            # Return modified outputs in the same format as the input
            return (inp,) + inputs[1:] if modified else inp

        return _steering_hook

    def _setup_steering_hooks(
        self, steering_function: SteeringFunction, steering_config: SteeringConfig
    ):
        """Set up hooks for applying steering.

        Args:
        ----
            steering_function: The steering function to apply.
            steering_config: The steering configuration.

        """
        self.steering_function = steering_function
        self.steering_config = steering_config
        self.modules = get_modules(self.model, steering_config.modules)
        self._debug(f"Setting up steering hooks for {len(self.modules)} modules")

        # Clear any existing hooks
        self.clear_hooks()

        # Initialize steering count (helper for keeping track of positions steered)
        self.steering_count = {}

        # Register hooks for each module
        for name, module in self.modules.items():
            self.steering_count[name] = 0
            self.hooks[name].append(module.register_forward_pre_hook(self._get_steering_hook(name)))
            self._debug(f"Registered steering hook for {name}")

    # API for caching and steering

    @contextmanager
    def caching(self, caching_config: CachingConfig):
        """Context manager for caching activations during model execution.

        Args:
        ----
            caching_config: The caching configuration.

        """
        self.start_caching(caching_config)
        try:
            yield self.cache
        finally:
            self.stop_caching()

    def start_caching(self, caching_config: CachingConfig):
        """Start caching activations.

        Args:
        ----
            caching_config: The caching configuration.

        """
        self._setup_caching_hooks(caching_config)

    def stop_caching(self):
        """Stop caching activations."""
        self.clear_hooks()
        self.caching_config = None

    @contextmanager
    def steering(self, steering_function, steering_config):
        """Context manager for applying steering during model execution.

        Args:
        ----
            steering_function: The steering function to apply.
            steering_config: The steering configuration.

        """
        assert steering_config.beta is not None, "Beta must be set in steering config"

        self.start_steering(steering_function, steering_config)
        try:
            yield
        finally:
            self.stop_steering()

    def start_steering(self, steering_function, steering_config):
        """Start applying steering.

        Args:
        ----
            steering_function: The steering function to apply.
            steering_config: The steering configuration.

        """
        self._setup_steering_hooks(steering_function, steering_config)

    def stop_steering(self):
        """Stop applying steering."""
        self.clear_hooks()
        self.steering_function = None
        self.steering_config = None
        self.steering_count = {}

    # Cache management
    def get_and_clear_cache(self) -> Dict[str, List[torch.Tensor]]:
        """Get the current cache and clear it.

        Returns
        -------
            A dictionary mapping module names to tensors (stacking the cached activations).

        """
        cache = dict(self.cache)

        # Clear the cache
        self.cache = defaultdict(list)

        return cache

    def clear_hooks(self):
        """Remove all hooks."""
        for hooks in self.hooks.values():
            for hook in hooks:
                hook.remove()
        self.hooks = defaultdict(list)


def main():
    """Demonstrate example usage of HookedLLM."""
    model_id = "EleutherAI/gpt-j-6b"
    model = HookedLLM(model_id)

    caching_cfg = CachingConfig(
        modules=ModuleSelection(
            layers=[11],
            layer_modules=[""],
            layer_path="transformer.h",
        ),
        tokens=TokenSelection(
            token=":",
        ),
    )
    with model.caching(caching_cfg):
        model.forward("Hello, world!")
    # Use the cached activations (uncomment to use)
    # act = model.get_and_clear_cache()


if __name__ == "__main__":
    main()
