import math

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3RotaryEmbedding, apply_rotary_pos_emb
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import Wav2Vec2BertSelfAttention
from transformers.pytorch_utils import Conv1D

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


class NativeW2VAttention(Wav2Vec2BertSelfAttention):
    def forward(self, hidden_states, attention_mask=None, relative_position_embeddings=None, **kwargs):
        shape = (*hidden_states.shape[:2], self.num_heads, self.head_size)
        q, k, v = (layer(hidden_states).reshape(shape).transpose(1, 2)
                   for layer in (self.linear_q, self.linear_k, self.linear_v))
        length = hidden_states.shape[1]
        positions = torch.arange(length, device=q.device)
        distance = (positions[None, :] - positions[:, None]).clamp(-self.left_max_position_embeddings,
                                                                 self.right_max_position_embeddings)
        relative = self.distance_embedding(distance + self.left_max_position_embeddings)
        bias = torch.einsum('bhld,lrd->bhlr', q, relative) * self.head_size**-0.5
        if attention_mask is not None:
            bias = bias + attention_mask
        return self.linear_out(attention(q, k, v, self.num_heads, bias, self.comfy_attention)), None


class NativeQwenAttention(Qwen3Attention):
    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None,
                cache_position=None, **kwargs):
        mm.throw_exception_if_processing_interrupted()
        shape=(*hidden_states.shape[:2],-1,self.head_dim)
        q=self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1,2)
        k=self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1,2)
        v=self.v_proj(hidden_states).view(shape).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,*position_embeddings)
        if past_key_values is not None:
            k,v=past_key_values.update(k,v,self.layer_idx,{'cache_position':cache_position})
        if attention_mask is None and q.shape[2]>1:
            keys=torch.arange(k.shape[2],device=q.device)
            queries=torch.arange(k.shape[2]-q.shape[2],k.shape[2],device=q.device)
            attention_mask=(keys[None,:]<=queries[:,None])[None,None]
        return self.o_proj(attention(q,k,v,self.config.num_attention_heads,attention_mask,self.comfy_attention)),None


class NativeGPTAttention(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.c_attn, self.c_proj = source.c_attn, source.c_proj
        self.heads, self.head_dim, self.split_size = source.num_heads, source.head_dim, source.split_size
        self.comfy_attention = ComfyAttention()

    def forward(self, x, cache=None):
        batch, length, _ = x.shape
        q, k, v = (part.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
                   for part in self.c_attn(x).split(self.split_size, dim=-1))
        offset = 0 if cache is None else cache[0].shape[2]
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=2), torch.cat((cache[1], v), dim=2)
        mask = None
        if length > 1:
            keys = torch.arange(offset + length, device=x.device)
            queries = torch.arange(offset, offset + length, device=x.device)
            mask = (keys[None, :] <= queries[:, None])[None, None]
        return self.c_proj(attention(q, k, v, self.heads, mask, self.comfy_attention)), (k, v)


class NativeGPT(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.h, self.ln_f, self.config = source.h, source.ln_f, source.config
        for block in self.h:
            block.attn = NativeGPTAttention(block.attn)

    def forward(self, inputs_embeds, past_key_values=None, use_cache=False, **kwargs):
        x = inputs_embeds
        cache = []
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
        return BaseModelOutputWithPast(last_hidden_state=self.ln_f(x), past_key_values=tuple(cache) if use_cache else None)


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
    if isinstance(module, GPT2Model):
        module = NativeGPT(module)
    elif isinstance(module, Wav2Vec2BertSelfAttention):
        module.__class__ = NativeW2VAttention
        module.comfy_attention = ComfyAttention()
    elif isinstance(module, Qwen3Attention):
        module.__class__ = NativeQwenAttention
        module.comfy_attention = ComfyAttention()
    elif isinstance(module, RelPositionMultiHeadedAttention):
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
    if isinstance(module,Qwen3RotaryEmbedding):
        with torch.device('cpu'):
            module=Qwen3RotaryEmbedding(module.config)
    for name, child in list(module.named_children()):
        replacement = None
        if isinstance(child, Conv1D):
            replacement = comfy.ops.manual_cast.Linear(child.weight.shape[0], child.weight.shape[1], device='meta')
            replacement.weight = nn.Parameter(child.weight.transpose(0, 1).contiguous())
            replacement.bias = child.bias
        elif type(child) in (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.LayerNorm, nn.GroupNorm):
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
