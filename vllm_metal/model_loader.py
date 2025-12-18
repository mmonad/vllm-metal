# SPDX-License-Identifier: Apache-2.0
"""Model loader for Metal backend."""

import torch

from vllm_metal._compat import VLLM_AVAILABLE, BaseModelLoader, init_logger
from vllm_metal.utils import get_optimal_dtype

logger = init_logger(__name__)


class MetalModelLoader(BaseModelLoader):
    """Model loader optimized for Metal backend.

    This loader handles model weight loading and conversion
    for execution on Apple Silicon.
    """

    def __init__(self, load_config):
        """Initialize the Metal model loader.

        Args:
            load_config: vLLM load configuration
        """
        super().__init__(load_config)
        self.load_config = load_config

    def _get_weights_iterator(self, model_name_or_path: str, revision: str):
        """Get an iterator over model weights.

        Args:
            model_name_or_path: Model name or path
            revision: Model revision

        Yields:
            Tuples of (name, tensor) for each weight
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required for weight loading")

        from vllm.model_executor.model_loader.weight_utils import (
            download_weights_from_hf,
            filter_duplicate_safetensors_files,
            pt_weights_iterator,
            safetensors_weights_iterator,
        )

        # Download weights if necessary
        hf_folder = download_weights_from_hf(
            model_name_or_path,
            self.load_config.download_dir,
            self.load_config.load_format,
            revision,
        )

        # Determine file format and get iterator
        hf_weights_files = filter_duplicate_safetensors_files(
            hf_folder,
            self.load_config.load_format,
        )

        if hf_weights_files:
            # Use safetensors if available
            return safetensors_weights_iterator(hf_weights_files)
        else:
            # Fall back to PyTorch format
            return pt_weights_iterator(hf_folder)

    def load_model(self, model_config, device_config, scheduler_config=None):
        """Load a model for Metal execution.

        Args:
            model_config: Model configuration
            device_config: Device configuration
            scheduler_config: Scheduler configuration

        Returns:
            Loaded model
        """
        if not VLLM_AVAILABLE:
            raise ImportError("vLLM is required for model loading")

        from vllm.model_executor.model_loader.loader import get_model

        # Ensure we're loading to Metal device
        device = torch.device("mps")  # PyTorch uses "mps" for Metal

        # Get optimal dtype for Metal
        target_dtype = get_optimal_dtype()
        if model_config.dtype != "auto":
            target_dtype = getattr(torch, model_config.dtype)

        logger.info(
            f"Loading model '{model_config.model}' "
            f"with dtype={target_dtype} on device={device}"
        )

        # Use vLLM's standard model loading
        model = get_model(
            model_config=model_config,
            device_config=device_config,
            load_config=self.load_config,
            scheduler_config=scheduler_config,
        )

        # Move to Metal if not already there
        if not str(model.device).startswith("mps"):  # PyTorch uses "mps" for Metal
            model = model.to(device=device, dtype=target_dtype)

        return model


def load_weights_to_metal(
    weights_iterator,
    model: torch.nn.Module,
    dtype: torch.dtype = torch.float16,
) -> None:
    """Load weights from an iterator into a model on Metal.

    Args:
        weights_iterator: Iterator yielding (name, tensor) tuples
        model: Model to load weights into
        dtype: Target dtype for weights
    """
    device = torch.device("mps")  # PyTorch uses "mps" for Metal
    state_dict = model.state_dict()

    for name, weight in weights_iterator:
        if name in state_dict:
            # Convert weight to target dtype and device
            param = state_dict[name]
            if weight.shape != param.shape:
                logger.warning(
                    f"Shape mismatch for {name}: "
                    f"expected {param.shape}, got {weight.shape}"
                )
                continue

            # Convert and copy weight
            weight = weight.to(dtype=dtype, device=device)
            param.copy_(weight)
        else:
            logger.debug(f"Skipping weight {name}: not in model state dict")


def convert_weights_for_metal(
    state_dict: dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """Convert a state dict for Metal execution.

    Args:
        state_dict: Model state dictionary
        dtype: Target dtype

    Returns:
        Converted state dictionary
    """
    converted = {}
    device = torch.device("mps")  # PyTorch uses "mps" for Metal

    for name, tensor in state_dict.items():
        # Convert to target dtype
        if tensor.dtype in (torch.float32, torch.float64):
            tensor = tensor.to(dtype=dtype)

        # Move to Metal
        tensor = tensor.to(device=device)

        converted[name] = tensor

    return converted


def estimate_model_memory(
    model_config,
    dtype: torch.dtype = torch.float16,
) -> int:
    """Estimate memory requirements for a model.

    Args:
        model_config: Model configuration
        dtype: Data type for weights

    Returns:
        Estimated memory in bytes
    """
    # Calculate parameter count
    hidden_size = model_config.get_hidden_size()
    num_layers = model_config.get_num_layers(parallel_config=None)
    vocab_size = model_config.get_vocab_size()

    # Rough estimation based on transformer architecture
    # This is a simplified estimate
    bytes_per_param = 2 if dtype in (torch.float16, torch.bfloat16) else 4

    # Embedding: vocab_size * hidden_size
    embedding_params = vocab_size * hidden_size

    # Per layer (rough estimate):
    # - QKV projection: 3 * hidden_size * hidden_size
    # - Output projection: hidden_size * hidden_size
    # - MLP: 2 * hidden_size * (4 * hidden_size)
    # - LayerNorms: 2 * hidden_size
    layer_params = (
        4 * hidden_size * hidden_size  # Attention
        + 8 * hidden_size * hidden_size  # MLP (assuming 4x expansion)
        + 2 * hidden_size  # LayerNorms
    )

    total_params = embedding_params + num_layers * layer_params

    return int(total_params * bytes_per_param)
