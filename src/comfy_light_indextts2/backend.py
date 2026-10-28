import base64
import copy
import json
import logging
import re
from dataclasses import fields

import torch
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm
import torchaudio.functional as AF
from torchaudio.compliance.kaldi import fbank
from tokenizers import Tokenizer

import comfy.latent_formats
import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.sd
from comfy.text_encoders.llama import Qwen3_06B, Qwen3_06BConfig
from comfy.text_encoders.spiece_tokenizer import SPieceTokenizer

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.s2mel.modules.bigvgan.bigvgan import BigVGAN
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.commons import MyModel
from indextts.utils.common import tokenize_by_CJK_char
from indextts.utils.config import AttrDict
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.maskgct.models.codec.kmeans.repcodec_model import RepCodec

from .native import native_modules
from .semantic import SemanticEncoder, reference_features


EMOTIONS = ('happy', 'angry', 'sad', 'fear', 'hate', 'low', 'surprise', 'neutral')
EMOTION_LABELS = (('喜','高兴','happy'),('怒','愤怒','angry'),('哀','悲伤','sad'),('惧','恐惧','afraid','fear'),
                 ('厌恶','反感','disgusted','hate'),('低落','melancholic','low'),('惊喜','惊讶','surprised','surprise'),
                 ('自然','calm','neutral'))


class MelLatent(comfy.latent_formats.LatentFormat):
    latent_channels = 80
    latent_dimensions = 1
    spacial_downscale_ratio = 1
    temporal_downscale_ratio = 256


class BundledTokenizer(TextTokenizer):
    def __init__(self, assets):
        self.normalizer = TextNormalizer()
        self.sp_model = SPieceTokenizer(base64.b64decode(assets['bpe_model_base64']), add_eos=False).tokenizer
        self.pre_tokenizers = [tokenize_by_CJK_char]


def mel_spectrogram(audio, config):
    params = config.s2mel.preprocess_params.spect_params
    n_fft, hop, win = params.n_fft, params.hop_length, params.win_length
    audio = F.pad(audio[:, None], ((n_fft-hop)//2, (n_fft-hop)//2), mode='reflect').squeeze(1)
    window = torch.hann_window(win, device=audio.device)
    spec = torch.stft(audio, n_fft, hop_length=hop, win_length=win, window=window,
                      center=False, return_complex=True)
    magnitude = torch.sqrt(spec.real.square() + spec.imag.square() + 1e-9)
    with torch.device(audio.device):
        bank = AF.melscale_fbanks(n_fft//2+1, params.fmin, 11025., params.n_mels, 22050,
                                  norm='slaney', mel_scale='slaney').transpose(0, 1)
    return torch.log(torch.clamp(bank @ magnitude, min=1e-5))


class IndexTTSModel(torch.nn.Module):
    def __init__(self, config, checkpoint_id, dtype):
        super().__init__()
        self.config, self.checkpoint_id, self.compute_dtype = config, checkpoint_id, dtype
        self.latent_format = MelLatent()
        with torch.device('meta'):
            self.gpt = UnifiedVoice(**config.gpt)
            self.s2mel = MyModel(config.s2mel, use_gpt_latent=True).models
            self.semantic_codec = RepCodec(cfg=config.semantic_codec)

    def get_dtype(self):
        return self.compute_dtype

    def process_latent_out(self, samples):
        return samples


class ReferenceEncoder(torch.nn.Module):
    def __init__(self, semantic_config):
        super().__init__()
        with torch.device('meta'):
            self.semantic_model = SemanticEncoder(AttrDict(semantic_config))
            self.campplus = CAMPPlus(feat_dim=80, embedding_size=192)
        self.register_buffer('mean', torch.empty(1024))
        self.register_buffer('std', torch.empty(1024))
        self.register_buffer('speaker_matrix', torch.empty(73, 192))
        self.register_buffer('emotion_matrix', torch.empty(73, 1280))

    def semantic_features(self, features, mask):
        return (self.semantic_model(features, mask)-self.mean)/self.std


class IndexTTSClip(comfy.sd.CLIP):
    def __init__(self, encoder, emotion, assets, config, checkpoint_id):
        self.cond_stage_model = encoder
        self.patcher = comfy.model_patcher.ModelPatcher(encoder, mm.text_encoder_device(), mm.text_encoder_offload_device())
        self.emotion_patcher = comfy.model_patcher.ModelPatcher(emotion, mm.text_encoder_device(), mm.text_encoder_offload_device())
        self.tokenizer = BundledTokenizer(assets)
        files = assets['files']
        self.emotion_tokenizer = Tokenizer.from_str(files['emotion/tokenizer.json'])
        self.emotion_eos = json.loads(files['emotion/generation_config.json'])['eos_token_id']
        self.config, self.checkpoint_id = config, checkpoint_id

    def clone(self, disable_dynamic=False):
        result = copy.copy(self)
        result.patcher = self.patcher.clone()
        result.emotion_patcher = self.emotion_patcher.clone()
        return result

    def emotion_from_text(self, text):
        patcher = self.emotion_patcher
        mm.load_models_gpu([patcher], memory_required=256*1024**2, force_full_load=True)
        prompt = '请对以下文本进行情感打分，给出喜、怒、哀、惧、低落、厌恶、自然、惊喜这八种情感的得分（0-1.2之间）。'
        tokenizer = self.emotion_tokenizer
        encoded = 'Human: '+prompt+'\n'+text+'<|endoftext|>\r\nAssistant:'
        inputs = torch.tensor([tokenizer.encode(encoded).ids], device=patcher.load_device)
        patcher.pre_run()
        try:
            mm.throw_exception_if_processing_interrupted()
            emotion = patcher.model
            output = emotion.generate(embeds=emotion.model.embed_tokens(inputs), do_sample=False, max_length=256,
                                      execution_dtype=emotion.dtype, stop_tokens=[self.emotion_eos])
            content = tokenizer.decode(output, skip_special_tokens=True)
        finally:
            patcher.cleanup()
        scores = {key:float(value) for key,value in re.findall(r'([\w]+)["\s]*[:：]["\s]*([+-]?\d+(?:\.\d+)?)', content)}
        values = [min(1.2,max(0.,next((scores[label] for label in names if label in scores),0.))) for names in EMOTION_LABELS]
        if not any(values):
            logging.warning('IndexTTS emotion text returned no scores; using neutral emotion.')
            values[-1] = 1.
        return values

    def encode(self, text, speaker_audio, emotion_audio=None, emotion_vector=None, emotion_text=None,
               emotion_alpha=1., max_text_tokens=120):
        if emotion_text is not None:
            emotion_vector = self.emotion_from_text(emotion_text.strip() or text)
        tokens = self.tokenizer.tokenize(text)
        segments = self.tokenizer.split_sentences(tokens, max_text_tokens)
        if not segments:
            raise ValueError('Enter text to synthesize.')
        patcher = self.patcher
        mm.load_models_gpu([patcher], memory_required=512*1024**2, force_full_load=True)
        device = patcher.load_device
        model = self.cond_stage_model

        def reference(audio, with_style):
            wave = audio['waveform'][0].mean(dim=0, keepdim=True).float().cpu()
            rate = audio['sample_rate']
            wave16 = AF.resample(wave, rate, 16000)
            features, mask = reference_features(wave16)
            mm.throw_exception_if_processing_interrupted()
            semantic = model.semantic_features(features.to(device),mask.to(device))
            if not with_style:
                return semantic
            wave22 = AF.resample(wave, rate, 22050).to(device)
            mel = mel_spectrogram(wave22, self.config)
            feat = fbank(wave16.to(device), num_mel_bins=80, dither=0, sample_frequency=16000)
            style = model.campplus((feat-feat.mean(dim=0,keepdim=True)).unsqueeze(0))
            return semantic, mel, style

        patcher.pre_run()
        try:
            speaker, mel, style = reference(speaker_audio, True)
            emotion = speaker if emotion_audio is None else reference(emotion_audio, False)
            data = {'checkpoint_id':self.checkpoint_id, 'speaker':speaker.cpu(), 'emotion':emotion.cpu(),
                    'style':style.cpu(), 'ref_mel':mel.cpu(), 'emotion_alpha':emotion_alpha}
            if emotion_vector is not None and any(emotion_vector):
                speakers = model.speaker_matrix.split(self.config.emo_num)
                emotions = model.emotion_matrix.split(self.config.emo_num)
                selected = torch.stack([group[F.cosine_similarity(style,bank,dim=1).argmax()]
                                        for bank,group in zip(speakers,emotions)])
                weights = torch.tensor(emotion_vector, device=device, dtype=torch.float32)
                data['emotion_vector'] = (weights[:,None]*selected).sum(0,keepdim=True).cpu()
                data['emotion_weight'] = sum(emotion_vector)
            return [[[torch.tensor([self.tokenizer.convert_tokens_to_ids(segment)],dtype=torch.long),
                      {'indextts':dict(data,segment_index=index)}]] for index,segment in enumerate(segments)]
        finally:
            patcher.cleanup()


class IndexTTSVAE(comfy.sd.VAE):
    def __init__(self, model, checkpoint_id):
        self.device, self.output_device = mm.vae_device(), mm.intermediate_device()
        self.vae_dtype = torch.float32
        self.audio_sample_rate = self.audio_sample_rate_output = 22050
        self.latent_channels, self.latent_dim, self.output_channels = 80, 1, 1
        self.downscale_ratio = self.upscale_ratio = 256
        self.first_stage_model = model
        self.checkpoint_id = checkpoint_id
        self.patcher = comfy.model_patcher.ModelPatcher(model, self.device, mm.vae_offload_device())

    def decode(self, samples_in, vae_options=None):
        mm.load_models_gpu([self.patcher], memory_required=512*1024**2, force_full_load=True)
        self.patcher.pre_run()
        try:
            mm.throw_exception_if_processing_interrupted()
            output = self.first_stage_model(samples_in.to(device=self.device,dtype=torch.float32))
            return output.movedim(1,-1).to(self.output_device)
        finally:
            self.patcher.cleanup()


def load_checkpoint(weights, metadata, dtype=None):
    if metadata.get('modelspec.architecture') != 'indextts' or metadata.get('indextts.format_version') != '1':
        raise ValueError('Select an IndexTTS checkpoint converted by this plugin.')
    if metadata.get('indextts.model_version') != '2.0':
        raise ValueError('This loader supports IndexTTS 2.0 checkpoints.')
    required=('indextts.config','indextts.semantic_config','indextts.feature_extractor_config',
              'indextts.vocoder_config','indextts.emotion_model_config','indextts.weights_sha256')
    missing=[key for key in required if key not in metadata]
    if missing:
        raise ValueError('Incomplete IndexTTS checkpoint metadata: '+', '.join(missing))
    config = AttrDict(json.loads(metadata['indextts.config']))
    checkpoint_id = metadata['indextts.weights_sha256']
    device = mm.get_torch_device()
    parameters=sum(value.numel() for key,value in weights.items() if key.startswith('model.diffusion_model.'))
    dtype = dtype or mm.unet_dtype(device=device,model_params=parameters,
                                  supported_dtypes=[torch.float16,torch.bfloat16,torch.float32])
    if device.type == 'cpu':
        dtype = torch.float32
    if dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise ValueError('IndexTTS supports float16, bfloat16 and float32. Select an explicit precision in its loader.')

    def take(prefix):
        return {key.removeprefix(prefix):weights.pop(key) for key in list(weights) if key.startswith(prefix)}

    network = IndexTTSModel(config,checkpoint_id,dtype)
    network_weights = take('model.diffusion_model.')
    # Original GPT-2 checkpoints store Conv1D matrices in [in, out] order.
    for key in network_weights:
        if key.startswith('gpt.gpt.h.') and key.endswith('.weight') and ('.attn.c_' in key or '.mlp.c_' in key):
            network_weights[key] = network_weights[key].transpose(0, 1).contiguous()
    network.load_state_dict(network_weights,strict=True,assign=True)
    for module in network.modules():
        if 'weight_g' in module._parameters:
            remove_weight_norm(module)
    network.gpt.to(dtype=dtype)
    network = native_modules(network).eval()
    model = comfy.model_patcher.ModelPatcher(network,device,mm.unet_offload_device())
    encoder = ReferenceEncoder(json.loads(metadata['indextts.semantic_config']))
    semantic_weights = take('cond_stage_model.semantic_model.')
    # Only hidden_states[17] is consumed; discard the training mask and later layers.
    semantic_weights.pop('masked_spec_embed')
    for key in list(semantic_weights):
        if key.startswith('encoder.layers.') and int(key.split('.')[2]) >= 17:
            del semantic_weights[key]
    encoder.semantic_model.load_state_dict(semantic_weights,strict=True,assign=True)
    encoder.campplus.load_state_dict(take('cond_stage_model.campplus.'),strict=True,assign=True)
    stats = take('cond_stage_model.semantic_stats.')
    encoder.mean, encoder.std = stats['mean'], stats['var'].sqrt()
    encoder.speaker_matrix = weights.pop('cond_stage_model.emotion.speaker_matrix')
    encoder.emotion_matrix = weights.pop('cond_stage_model.emotion.emotion_matrix')
    encoder = native_modules(encoder).eval()
    assets = json.loads(bytes(weights.pop('cond_stage_model.indextts_assets').numpy()).decode('utf-8'))
    emotion_weights = take('cond_stage_model.emotion_model.')
    emotion_config = json.loads(metadata['indextts.emotion_model_config'])
    config_keys = {field.name for field in fields(Qwen3_06BConfig)}
    emotion = Qwen3_06B({key:value for key,value in emotion_config.items() if key in config_keys},
        dtype=emotion_weights['model.embed_tokens.weight'].dtype, device='meta', operations=comfy.ops.manual_cast)
    emotion.load_state_dict(emotion_weights,strict=True,assign=True)
    emotion.eval()
    with torch.device('meta'):
        vocoder = BigVGAN(AttrDict(json.loads(metadata['indextts.vocoder_config'])))
    vocoder.load_state_dict(take('first_stage_model.'),strict=True,assign=True)
    vocoder.remove_weight_norm()
    vocoder = native_modules(vocoder).eval()
    if weights:
        raise ValueError(f'Unrecognized IndexTTS checkpoint weights: {next(iter(weights))}')
    clip = IndexTTSClip(encoder,emotion,assets,config,checkpoint_id)
    return model,clip,IndexTTSVAE(vocoder,checkpoint_id)
