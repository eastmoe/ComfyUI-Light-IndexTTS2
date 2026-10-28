import math
from types import SimpleNamespace

import torch
from torch import nn

import comfy.model_management as mm
import comfy.model_prefetch
from comfy.ops import manual_cast as ops
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_for_device


class GPTAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.c_attn = ops.Linear(width, 3 * width)
        self.c_proj = ops.Linear(width, width)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, cache):
        batch, length, width = x.shape
        q, k, v = (part.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
                   for part in self.c_attn(x).split(width, dim=-1))
        offset = 0 if cache is None else cache[0].shape[2]
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        mask = None
        if length > 1:
            keys = torch.arange(offset + length, device=x.device)
            queries = torch.arange(offset, offset + length, device=x.device)
            mask = (keys[None, :] <= queries[:, None])[None, None]
        function = optimized_attention_for_device(x.device, mask=mask is not None)
        result = function(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
                          self.heads, mask=mask, skip_reshape=True, enable_gqa=True,
                          preferred_attention=self.comfy_attention)
        return self.c_proj(result), (k, v)


class GPTMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c_fc = ops.Linear(width, width * 4)
        self.c_proj = ops.Linear(width * 4, width)

    def forward(self, x):
        x = self.c_fc(x)
        x = 0.5 * x * (1. + torch.tanh(math.sqrt(2. / math.pi) * (x + 0.044715 * x.pow(3.))))
        return self.c_proj(x)


class GPTBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.ln_1 = ops.LayerNorm(width, eps=1e-5)
        self.attn = GPTAttention(width, heads)
        self.ln_2 = ops.LayerNorm(width, eps=1e-5)
        self.mlp = GPTMLP(width)


class GPT(nn.Module):
    def __init__(self, layers, width, heads):
        super().__init__()
        self.h = nn.ModuleList([GPTBlock(width, heads) for _ in range(layers)])
        self.ln_f = ops.LayerNorm(width, eps=1e-5)

    def forward(self, inputs_embeds, past_key_values=None, use_cache=False, **kwargs):
        x, cache = inputs_embeds, []
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.h), x.device, {'prefetch_dynamic_vbars': True})
        for index, block in enumerate(self.h):
            mm.throw_exception_if_processing_interrupted()
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, block, x.dtype, malloc_scope='block')
            attended, kv = block.attn(block.ln_1(x), None if past_key_values is None else past_key_values[index])
            x = x + attended
            x = x + block.mlp(block.ln_2(x))
            if use_cache:
                cache.append(kv)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, x.dtype, malloc_scope='block')
        return SimpleNamespace(last_hidden_state=self.ln_f(x), past_key_values=tuple(cache) if use_cache else None)
