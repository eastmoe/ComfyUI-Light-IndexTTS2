"""Fallback attention-mask helpers for Transformers versions that move internals."""

from __future__ import annotations

import torch


def _expand_2d_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int | None = None) -> torch.Tensor:
    batch_size, src_len = mask.shape
    tgt_len = src_len if tgt_len is None else tgt_len
    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, src_len).to(dtype)
    return (1.0 - expanded_mask) * torch.finfo(dtype).min


def _prepare_4d_attention_mask_for_sdpa(
    mask: torch.Tensor | None,
    dtype: torch.dtype,
    tgt_len: int | None = None,
) -> torch.Tensor | None:
    if mask is None:
        return None
    if mask.dim() == 4:
        return mask
    if mask.dim() != 2:
        raise ValueError(f"Expected a 2D or 4D attention mask, got shape {tuple(mask.shape)}.")
    if torch.all(mask == 1):
        return None
    return _expand_2d_mask(mask, dtype, tgt_len)


def _prepare_4d_causal_attention_mask_for_sdpa(
    attention_mask: torch.Tensor | None,
    input_shape: tuple[int, int],
    inputs_embeds: torch.Tensor,
    past_key_values_length: int = 0,
    sliding_window: int | None = None,
) -> torch.Tensor | None:
    batch_size, query_len = input_shape
    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    key_value_len = query_len + past_key_values_length

    if attention_mask is None and query_len > 1:
        return None
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask

    min_value = torch.finfo(dtype).min
    mask = torch.full((query_len, key_value_len), min_value, device=device, dtype=dtype)
    past = torch.arange(key_value_len, device=device)
    current = torch.arange(past_key_values_length, key_value_len, device=device)
    causal = past.unsqueeze(0) <= current.unsqueeze(1)

    if sliding_window is not None:
        causal = causal & (past.unsqueeze(0) > (current - sliding_window).unsqueeze(1))

    mask = mask.masked_fill(causal, 0.0)
    mask = mask[None, None, :, :].expand(batch_size, 1, query_len, key_value_len)

    if attention_mask is not None:
        padding_mask = _expand_2d_mask(attention_mask, dtype, query_len)
        mask = mask + padding_mask

    return mask
