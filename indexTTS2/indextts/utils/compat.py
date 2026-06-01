"""Small compatibility helpers for supported dependency version ranges."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


def torch_load(*args: Any, weights_only: bool = False, **kwargs: Any) -> Any:
    """Load checkpoints with stable behavior across PyTorch 2.1 through 2.11."""

    try:
        return torch.load(*args, weights_only=weights_only, **kwargs)
    except TypeError as exc:
        if "weights_only" not in str(exc):
            raise
        return torch.load(*args, **kwargs)


def get_device_map(n_layers: int, devices: Sequence[int]) -> dict[int, list[int]]:
    """Fallback for ``transformers.utils.model_parallel_utils.get_device_map``."""

    devices = list(devices)
    if not devices:
        return {}

    layers = list(range(n_layers))
    layers_per_device = (n_layers + len(devices) - 1) // len(devices)
    return {
        device: layers[index * layers_per_device : (index + 1) * layers_per_device]
        for index, device in enumerate(devices)
        if layers[index * layers_per_device : (index + 1) * layers_per_device]
    }


def assert_device_map(device_map: Mapping[int, Sequence[int]], n_layers: int) -> None:
    """Fallback for ``transformers.utils.model_parallel_utils.assert_device_map``."""

    layers = sorted(layer for layers_on_device in device_map.values() for layer in layers_on_device)
    expected = list(range(n_layers))
    if layers != expected:
        raise ValueError(f"The device_map does not cover all layers. Got {layers}, expected {expected}.")


def isin_mps_friendly(elements: torch.Tensor, test_elements: torch.Tensor | int | float) -> torch.Tensor:
    """Use ``torch.isin`` while keeping compatibility with older MPS builds."""

    if not isinstance(test_elements, torch.Tensor):
        test_elements = torch.tensor(test_elements, device=elements.device)

    if elements.device.type == "mps":
        return (elements[..., None] == test_elements.reshape(-1)).any(-1)
    return torch.isin(elements, test_elements)
