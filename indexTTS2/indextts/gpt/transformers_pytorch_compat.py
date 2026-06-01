from __future__ import annotations

from urllib.parse import urlparse

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D, prune_linear_layer

FLAX_WEIGHTS_NAME = "flax_model.msgpack"
TF2_WEIGHTS_NAME = "tf_model.h5"
TF_WEIGHTS_NAME = "model.ckpt"


def is_remote_url(url_or_filename: str) -> bool:
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def is_safetensors_available() -> bool:
    try:
        import safetensors  # noqa: F401
    except Exception:
        return False
    return True


try:
    from transformers.utils.hub import is_offline_mode
except (ImportError, AttributeError):
    def is_offline_mode() -> bool:
        return False


def download_url(url: str, proxies: dict | None = None) -> str:
    import tempfile
    from pathlib import Path

    import requests

    response = requests.get(url, proxies=proxies, stream=True)
    response.raise_for_status()
    parsed_name = Path(urlparse(url).path).name or "downloaded_model_file"
    output_path = Path(tempfile.mkdtemp()) / parsed_name
    with output_path.open("wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
    return str(output_path)


def find_pruneable_heads_and_indices(
    heads: list[int] | set[int],
    n_heads: int,
    head_size: int,
    already_pruned_heads: set[int],
) -> tuple[set[int], torch.LongTensor]:
    heads = set(heads) - already_pruned_heads
    mask = torch.ones(n_heads, head_size)
    for head in heads:
        head = head - sum(1 if pruned_head < head else 0 for pruned_head in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index


def prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, dim: int = 1) -> Conv1D:
    index = index.to(layer.weight.device)
    weight = layer.weight.index_select(dim, index).detach().clone()
    if dim == 0:
        bias = layer.bias.detach().clone()
    else:
        bias = layer.bias[index].detach().clone()

    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(weight.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(bias.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


def prune_layer(layer: nn.Module, index: torch.LongTensor, dim: int | None = None) -> nn.Module:
    if isinstance(layer, nn.Linear):
        return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    if isinstance(layer, Conv1D):
        return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    raise ValueError(f"Cannot prune layer of class {layer.__class__}")
