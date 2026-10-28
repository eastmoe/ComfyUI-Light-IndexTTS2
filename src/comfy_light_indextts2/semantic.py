import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import comfy.model_management as mm
import comfy.model_prefetch
from comfy.ops import manual_cast as ops
from comfy.ldm.modules.attention import ComfyAttention

from .native import attention


def reference_features(wave):
    """Kaldi log-mel analysis with the checkpoint's original FFT and CMVN precision."""
    wave = wave.reshape(-1).float().numpy() * 32768.
    if wave.size < 560:
        wave = np.pad(wave, (0, 560 - wave.size))
    frames = np.lib.stride_tricks.sliding_window_view(wave.astype(np.float64), 400)[::160].copy()
    frames -= frames.mean(-1, keepdims=True)
    frames[:, 1:] -= .97 * frames[:, :-1]
    frames[:, 0] *= .03
    spectrum = np.fft.rfft(frames * (np.hanning(400) ** .85), n=512).astype(np.complex64)
    power = np.abs(spectrum, dtype=np.float64) ** 2
    mel = 1127. * np.log(1. + np.arange(257) * (16000. / 512) / 700.)
    edges = np.linspace(1127. * np.log(1. + 20. / 700.), 1127. * np.log(1. + 8000. / 700.), 82)
    slopes = edges[None, :] - mel[:, None]
    widths = np.diff(edges)
    bank = np.maximum(0., np.minimum(-slopes[:, :-2] / widths[:-1], slopes[:, 2:] / widths[1:]))
    features = np.log(np.maximum(1.192092955078125e-7, bank.T @ power.T)).astype(np.float32).T
    features = (features - features.mean(0)) / np.sqrt(features.var(0, ddof=1) + 1e-7)
    length = features.shape[0]
    if length % 2:
        features = np.pad(features, ((0, 1), (0, 0)), constant_values=1.)
    features = torch.from_numpy(features.reshape(1, -1, 160))
    mask = torch.arange(features.shape[1])[None] * 2 + 1 < length
    return features, mask.long()


class FeatureProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = ops.LayerNorm(config.feature_projection_input_dim, eps=config.layer_norm_eps)
        self.projection = ops.Linear(config.feature_projection_input_dim, config.hidden_size)

    def forward(self, x):
        return self.projection(self.layer_norm(x))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dense = ops.Linear(config.hidden_size, config.intermediate_size)
        self.output_dense = ops.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.output_dense(F.silu(self.intermediate_dense(x)))


class RelativeAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.num_heads, self.head_size = config.num_attention_heads, width // config.num_attention_heads
        self.left, self.right = config.left_max_position_embeddings, config.right_max_position_embeddings
        self.linear_q, self.linear_k, self.linear_v = (ops.Linear(width, width) for _ in range(3))
        self.linear_out = ops.Linear(width, width)
        self.distance_embedding = ops.Embedding(self.left + self.right + 1, self.head_size)
        self.comfy_attention = ComfyAttention()

    def forward(self, x, mask):
        shape = (*x.shape[:2], self.num_heads, self.head_size)
        q, k, v = (layer(x).reshape(shape).transpose(1, 2) for layer in (self.linear_q, self.linear_k, self.linear_v))
        positions = torch.arange(x.shape[1], device=x.device)
        distance = (positions[None, :] - positions[:, None]).clamp(-self.left, self.right)
        relative = self.distance_embedding(distance + self.left)
        bias = torch.einsum('bhld,lrd->bhlr', q, relative) * self.head_size**-.5 + mask
        return self.linear_out(attention(q, k, v, self.num_heads, bias, self.comfy_attention))


class Convolution(nn.Module):
    def __init__(self, config):
        super().__init__()
        width = config.hidden_size
        self.layer_norm = ops.LayerNorm(width, eps=config.layer_norm_eps)
        self.pointwise_conv1 = ops.Conv1d(width, width * 2, 1, bias=False)
        self.depthwise_conv = ops.Conv1d(width, width, config.conv_depthwise_kernel_size, groups=width, bias=False)
        self.depthwise_layer_norm = ops.LayerNorm(width, eps=config.layer_norm_eps)
        self.pointwise_conv2 = ops.Conv1d(width, width, 1, bias=False)

    def forward(self, x, valid):
        x = self.layer_norm(x).masked_fill(~valid[:, :, None], 0.).transpose(1, 2)
        x = F.glu(self.pointwise_conv1(x), dim=1)
        x = self.depthwise_conv(F.pad(x, (self.depthwise_conv.kernel_size[0] - 1, 0)))
        x = self.depthwise_layer_norm(x.transpose(1, 2)).transpose(1, 2)
        return self.pointwise_conv2(F.silu(x)).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn1_layer_norm = ops.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn1 = FeedForward(config)
        self.self_attn_layer_norm = ops.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = RelativeAttention(config)
        self.conv_module = Convolution(config)
        self.ffn2_layer_norm = ops.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn2 = FeedForward(config)
        self.final_layer_norm = ops.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, mask, valid):
        x = self.ffn1(self.ffn1_layer_norm(x)) * .5 + x
        x = self.self_attn(self.self_attn_layer_norm(x), mask) + x
        x = x + self.conv_module(x, valid)
        x = self.ffn2(self.ffn2_layer_norm(x)) * .5 + x
        return self.final_layer_norm(x)


class SemanticEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.position_embeddings_type != 'relative_key' or config.hidden_act != 'swish':
            raise ValueError('Unsupported IndexTTS semantic encoder configuration.')
        self.feature_projection = FeatureProjection(config)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList([EncoderLayer(config) for _ in range(17)])

    def forward(self, features, mask):
        valid = mask.bool()
        x = self.feature_projection(features).masked_fill(~valid[:, :, None], 0.)
        bias = torch.zeros((x.shape[0], 1, 1, x.shape[1]), device=x.device, dtype=x.dtype)
        bias = bias.masked_fill(~valid[:, None, None, :], -torch.inf)
        queue = comfy.model_prefetch.make_prefetch_queue(list(self.encoder.layers), x.device, {'prefetch_dynamic_vbars': True})
        for layer in self.encoder.layers:
            mm.throw_exception_if_processing_interrupted()
            comfy.model_prefetch.prefetch_queue_pop(queue, x.device, layer, x.dtype, malloc_scope='block')
            x = layer(x, bias, valid)
        comfy.model_prefetch.prefetch_queue_pop(queue, x.device, None, x.dtype, malloc_scope='block')
        return x
