import math

import torch
from torch import nn

import comfy.model_management as mm
import comfy.model_prefetch
import comfy.ops
from comfy.ldm.modules.attention import AttentionTensorContainer, ComfyAttention, optimized_attention_for_device

from indextts.gpt.conformer.attention import RelPositionMultiHeadedAttention
from indextts.gpt.perceiver import Attend
from indextts.s2mel.modules.gpt_fast.model import Attention, Transformer, apply_rotary_emb, precompute_freqs_cis


def attention(q, k, v, heads, mask, preferred):
    function = optimized_attention_for_device(q.device, mask=mask is not None)
    return function(AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v),
                    heads, mask=mask, skip_reshape=True, enable_gqa=True, preferred_attention=preferred)


class NativeAttend(Attend):
    def forward(self, q, k, v, mask=None):
        if k.ndim == 3:
            k, v = k.unsqueeze(1), v.unsqueeze(1)
        if mask is not None:
            mask = mask[:, None, None, :]
        result = attention(q, k, v, q.shape[1], mask, self.comfy_attention)
        return result.reshape(q.shape[0], q.shape[2], q.shape[1], q.shape[3]).transpose(1, 2)


class NativeConformerAttention(RelPositionMultiHeadedAttention):
    def forward(self, query, key, value, mask=None, pos_emb=None, cache=None):
        q, k, v = self.forward_qkv(query, key, value)
        if cache is not None and cache.shape[0]:
            old_k, old_v = cache.chunk(2, dim=-1)
            k, v = torch.cat((old_k, k), dim=2), torch.cat((old_v, v), dim=2)
        positions = self.linear_pos(pos_emb).reshape(pos_emb.shape[0], -1, self.h, self.d_k).transpose(1, 2)
        bias_v = comfy.ops.cast_to_input(self.pos_bias_v, q)
        bias_u = comfy.ops.cast_to_input(self.pos_bias_u, q)
        bias = torch.matmul(q + bias_v[None, :, None], positions.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None and mask.shape[-1]:
            bias = bias.masked_fill(~mask[:, None, :, :k.shape[2]].bool(), -torch.inf)
        output = attention(q + bias_u[None, :, None], k, v, self.h, bias, self.comfy_attention)
        return self.linear_out(output), torch.cat((k, v), dim=-1)


class NativeFlowAttention(Attention):
    def forward(self, x, freqs_cis, mask, input_pos=None, context=None, context_freqs_cis=None):
        batch, length, _ = x.shape
        q, k, v = self.wqkv(x).chunk(3, dim=-1)
        q, k, v = (part.reshape(batch, length, self.n_head, self.head_dim) for part in (q, k, v))
        q = apply_rotary_emb(q, freqs_cis).transpose(1, 2)
        k = apply_rotary_emb(k, freqs_cis).transpose(1, 2)
        return self.wo(attention(q, k, v.transpose(1, 2), self.n_head, mask, self.comfy_attention))


class NativeFlowStack(Transformer):
    def forward(self, x, c, input_pos=None, mask=None, context=None, context_input_pos=None, cross_attention_mask=None):
        with torch.device(x.device):
            frequencies = precompute_freqs_cis(x.shape[1], self.config.head_dim, self.config.rope_base,
                                               self.norm.project_layer.weight.dtype)
        skips = []
        middle = len(self.layers) // 2
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.layers), x.device, {'prefetch_dynamic_vbars': True})
        for index, layer in enumerate(self.layers):
            mm.throw_exception_if_processing_interrupted()
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, malloc_scope='block')
            skip = skips.pop() if self.config.uvit_skip_connection and index > middle else None
            x = layer(x, c, input_pos, frequencies, mask, context, None, cross_attention_mask, skip)
            if self.config.uvit_skip_connection and index < middle:
                skips.append(x)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, x.dtype, malloc_scope='block')
        return self.norm(x, c)


def native_modules(module):
    if isinstance(module, RelPositionMultiHeadedAttention):
        module.__class__ = NativeConformerAttention
        module.comfy_attention = ComfyAttention()
    elif isinstance(module, Attend):
        module.__class__ = NativeAttend
        module.comfy_attention = ComfyAttention()
    elif isinstance(module, Attention):
        module.__class__ = NativeFlowAttention
        module.comfy_attention = ComfyAttention()
    elif isinstance(module, Transformer):
        module.__class__ = NativeFlowStack
    for name, child in list(module.named_children()):
        replacement = None
        if type(child) in (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.LayerNorm, nn.GroupNorm):
            operation = getattr(comfy.ops.manual_cast, type(child).__name__)
            if isinstance(child, nn.Linear):
                replacement = operation(child.in_features, child.out_features, bias=child.bias is not None, device='meta')
            elif isinstance(child, nn.Embedding):
                replacement = operation(child.num_embeddings, child.embedding_dim, padding_idx=child.padding_idx, device='meta')
            elif isinstance(child, nn.LayerNorm):
                replacement = operation(child.normalized_shape, eps=child.eps, elementwise_affine=child.elementwise_affine, device='meta')
            elif isinstance(child, nn.GroupNorm):
                replacement = operation(child.num_groups, child.num_channels, eps=child.eps, affine=child.affine, device='meta')
            else:
                options = {'output_padding':child.output_padding} if isinstance(child,nn.ConvTranspose1d) else {}
                replacement = operation(child.in_channels,child.out_channels,child.kernel_size,stride=child.stride,
                    padding=child.padding,dilation=child.dilation,groups=child.groups,bias=child.bias is not None,
                    padding_mode=child.padding_mode,device='meta',**options)
            for key,value in child._parameters.items():
                setattr(replacement,key,value)
        elif isinstance(child, nn.Dropout):
            replacement = nn.Identity()
        setattr(module, name, replacement if replacement is not None else native_modules(child))
    return module
