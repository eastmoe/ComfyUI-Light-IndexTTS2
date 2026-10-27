import math

import torch
from safetensors import safe_open

import comfy.model_management as mm
import comfy.utils
import folder_paths
from comfy_api.latest import io

from .backend import EMOTIONS, IndexTTSClip, load_checkpoint
from .sampling import IndexTTSGuider, IndexTTSRandomSource, IndexTTSSampler

CATEGORY='eastmoe/IndexTTS'
PRECISIONS={'auto':None,'float16':torch.float16,'bfloat16':torch.bfloat16,'float32':torch.float32}


def condition_inputs(speaker_optional=False):
    return [io.Clip.Input('clip',tooltip='CLIP from the IndexTTS Checkpoint Loader; includes text, reference audio and emotion encoders.'),io.String.Input('text',multiline=True,dynamic_prompts=True,tooltip='Text to synthesize. Long text is split into segments and processed in order.'),
            io.Audio.Input('speaker_audio',optional=speaker_optional,tooltip='Reference voice for speaker identity. In emotion audio mode, leaving this empty uses emotion_audio as the voice reference.'),
            io.Int.Input('max_text_tokens',default=120,min=8,max=600,advanced=True,
                         tooltip='Maximum text tokens per segment. Each segment is sampled and decoded separately; join the audio afterward.')]


def conditioning(clip,text,speaker_audio,max_text_tokens,**options):
    if not isinstance(clip,IndexTTSClip):
        raise ValueError('Connect CLIP from the IndexTTS Checkpoint Loader.')
    return io.NodeOutput(clip.encode(text,speaker_audio,max_text_tokens=max_text_tokens,**options))


class IndexTTSCheckpointLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        names=[name for name in folder_paths.get_filename_list('checkpoints') if name.lower().endswith('.safetensors')]
        return io.Schema(node_id='IndexTTSCheckpointLoader',display_name='IndexTTS Checkpoint Loader',category=CATEGORY,
            description='Load an offline IndexTTS checkpoint with its text/audio encoders and BigVGAN.',
            inputs=[io.Combo.Input('ckpt_name',options=names,tooltip='IndexTTS safetensors checkpoint in ComfyUI/models/checkpoints.'),io.Combo.Input('precision',options=list(PRECISIONS),default='auto',tooltip='Model compute precision. auto follows the ComfyUI device policy.')],
            outputs=[io.Model.Output(tooltip='Autoregressive GPT, semantic codec and CFM model for IndexTTS Guider.'),io.Clip.Output(tooltip='CLIP from the IndexTTS Checkpoint Loader; includes text, reference audio and emotion encoders.'),io.Vae.Output(tooltip='BigVGAN audio decoder for the native VAE Decode Audio node.')])

    @classmethod
    def execute(cls,ckpt_name,precision='auto'):
        if precision not in PRECISIONS:
            raise ValueError('Unsupported IndexTTS precision.')
        path=folder_paths.get_full_path_or_raise('checkpoints',ckpt_name)
        with safe_open(path,framework='pt',device='cpu') as checkpoint:
            if (checkpoint.metadata() or {}).get('modelspec.architecture')!='indextts':
                raise ValueError('Select an IndexTTS checkpoint converted by this plugin.')
        weights,metadata=comfy.utils.load_torch_file(path,return_metadata=True)
        return io.NodeOutput(*load_checkpoint(weights,metadata,PRECISIONS[precision]))


class IndexTTSSpeakerConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='IndexTTSSpeakerConditioning',display_name='IndexTTS Speaker Conditioning',category=CATEGORY+'/conditioning',
            description='Use the reference audio for speaker identity and emotion. Each text segment is sampled separately.',
            inputs=condition_inputs(),outputs=[io.Conditioning.Output(is_output_list=True,tooltip='One conditioning item per text segment; ComfyUI processes the list in order.')])

    @classmethod
    def execute(cls,clip,text,speaker_audio,max_text_tokens=120):
        return conditioning(clip,text,speaker_audio,max_text_tokens)


class IndexTTSEmotionAudioConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='IndexTTSEmotionAudioConditioning',display_name='IndexTTS Emotion Audio Conditioning',category=CATEGORY+'/conditioning',
            description='Use separate reference audio for speaker identity and emotional delivery.',
            inputs=condition_inputs(True)+[io.Audio.Input('emotion_audio',tooltip='Reference audio that supplies emotional delivery independently of speaker identity.'),io.Float.Input('emotion_alpha',default=1.,min=0.,max=2.,step=.05,tooltip='Interpolation from the speaker reference emotion (0) to the emotion audio (1); values above 1 strengthen the change.')],
            outputs=[io.Conditioning.Output(is_output_list=True,tooltip='One conditioning item per text segment; ComfyUI processes the list in order.')])

    @classmethod
    def execute(cls,clip,text,emotion_audio,speaker_audio=None,emotion_alpha=1.,max_text_tokens=120):
        return conditioning(clip,text,speaker_audio if speaker_audio is not None else emotion_audio,max_text_tokens,
                            emotion_audio=emotion_audio,emotion_alpha=emotion_alpha)


class IndexTTSEmotionVectorConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='IndexTTSEmotionVectorConditioning',display_name='IndexTTS Emotion Vector Conditioning',category=CATEGORY+'/conditioning',
            description='Select one voice-matched representative per emotion group, then mix the eight emotion weights.',
            inputs=condition_inputs()+[io.Float.Input(name,default=0.,min=0.,max=1.2,step=.05,tooltip=f'Weight for {name} emotion. 0 removes its contribution; the eight weights mix voice-matched emotion representatives.') for name in EMOTIONS],
            outputs=[io.Conditioning.Output(is_output_list=True,tooltip='One conditioning item per text segment; ComfyUI processes the list in order.')])

    @classmethod
    def execute(cls,clip,text,speaker_audio,max_text_tokens=120,**weights):
        return conditioning(clip,text,speaker_audio,max_text_tokens,emotion_vector=[weights.get(name,0.) for name in EMOTIONS])


class IndexTTSEmotionTextConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='IndexTTSEmotionTextConditioning',display_name='IndexTTS Emotion Text Conditioning',category=CATEGORY+'/conditioning',
            description='Analyze emotion text with the bundled Qwen model; an empty emotion prompt uses the speech text.',
            inputs=condition_inputs()+[io.String.Input('emotion_text',default='',multiline=True,tooltip='Describe the desired emotion. Empty text analyzes the speech text instead.')],
            outputs=[io.Conditioning.Output(is_output_list=True,tooltip='One conditioning item per text segment; ComfyUI processes the list in order.')])

    @classmethod
    def execute(cls,clip,text,speaker_audio,emotion_text='',max_text_tokens=120):
        return conditioning(clip,text,speaker_audio,max_text_tokens,emotion_text=emotion_text)


class IndexTTSGuiderNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='IndexTTSGuider',display_name='IndexTTS Guider',category=CATEGORY+'/sampling',
            inputs=[io.Model.Input('model',tooltip='MODEL from the IndexTTS Checkpoint Loader.'),io.Conditioning.Input('conditioning',tooltip='Text, voice and emotion conditions for one segment, supplied by an IndexTTS conditioning node.')],outputs=[io.Guider.Output(tooltip='IndexTTS model and conditions for SamplerCustomAdvanced.')])

    @classmethod
    def execute(cls,model,conditioning):
        return io.NodeOutput(IndexTTSGuider(model,conditioning))


class IndexTTSSamplerNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='IndexTTSSampler',display_name='IndexTTS AR / CFM Sampler',category=CATEGORY+'/sampling',
            description='Generate semantic tokens, then integrate S2Mel using the reference CFM Euler algorithm.',
            inputs=[io.Int.Input('max_tokens',default=1500,min=1,max=1815,tooltip='Maximum autoregressive semantic tokens per segment. EOS can end generation earlier; mel capacity also limits output length.'),io.Boolean.Input('do_sample',default=True,tooltip='Enable random token sampling. Disable for greedy decoding with one beam or deterministic beam search with multiple beams.'),
                io.Float.Input('temperature',default=.8,min=.01,max=3.,step=.05,tooltip='Token sampling temperature; higher values increase randomness. Used when do_sample is enabled.'),io.Int.Input('top_k',default=30,min=0,max=8194,tooltip='Keep the highest-scoring k tokens for random sampling. 0 disables this filter.'),
                io.Float.Input('top_p',default=.8,min=.01,max=1.,step=.01,tooltip='Keep the smallest token set reaching this cumulative probability for random sampling. 1 disables this filter.'),
                io.Float.Input('repetition_penalty',default=10.,min=.01,max=30.,step=.1,tooltip='Penalty applied to repeated semantic tokens. 1 disables the penalty; values above 1 discourage repetition.'),
                io.Int.Input('num_beams',default=1,min=1,max=16,advanced=True,tooltip='Number of beam search candidates. 1 uses ordinary greedy or random decoding.'),
                io.Float.Input('length_penalty',default=0.,min=-5.,max=5.,step=.1,advanced=True,tooltip='Beam search score normalization by sequence length. Positive values favor longer sequences; negative values favor shorter ones.'),
                io.Float.Input('typical_mass',default=1.,min=.01,max=1.,step=.01,advanced=True,tooltip='Typical sampling probability mass. Values below 1 filter candidate tokens in both greedy and random decoding; 1 disables it.'),
                io.Float.Input('cfm_cfg',default=.7,min=0.,max=5.,step=.05,
                               tooltip='CFM guidance strength. 0.7 corresponds to conventional CFG 1.7; 0 uses a single conditional prediction.'),
                io.Float.Input('cfm_temperature',default=1.,min=0.,max=3.,step=.05,tooltip='Scale of the initial CFM Gaussian noise. 0 starts from zero noise.'),
                io.Float.Input('length_scale',default=1.72,min=.1,max=5.,step=.01,advanced=True,tooltip='Mel frames per generated semantic token. The reference value is 1.72; larger values lengthen speech.'),
                io.Float.Input('target_duration',default=0.,min=0.,max=36.,step=.1,advanced=True,
                               tooltip='GPT duration condition in seconds per segment. 0 disables it; EOS and semantic length determine actual duration.')],
            outputs=[io.Sampler.Output(tooltip='Autoregressive token generation and CFM Euler integration for SamplerCustomAdvanced.')])

    @classmethod
    def execute(cls,**options):
        return io.NodeOutput(IndexTTSSampler(**options))


class IndexTTSRandomSourceNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='IndexTTSRandomSource',display_name='IndexTTS Random Source',category=CATEGORY+'/sampling',
            description='Seed for token sampling and CFM Gaussian noise, created after the generated length is known.',
            inputs=[io.Int.Input('seed',default=0,min=0,max=0xffffffffffffffff,control_after_generate=True,tooltip='Seed for autoregressive sampling and CFM Gaussian noise. Segments and batch items use separate seed offsets.')],outputs=[io.Noise.Output(tooltip='Random source for the noise input of SamplerCustomAdvanced.')])

    @classmethod
    def execute(cls,seed):
        return io.NodeOutput(IndexTTSRandomSource(seed))


class IndexTTSCFMSchedule(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='IndexTTSCFMSchedule',display_name='IndexTTS CFM Schedule',category=CATEGORY+'/sampling',
            description='CFM integration schedule: sigma=1-t. linear follows the reference timestep grid.',
            inputs=[io.Int.Input('steps',default=25,min=1,max=1000,tooltip='Number of CFM Euler integration steps. More steps increase sampling time.'),
                    io.Combo.Input('schedule',options=['linear','cosine'],default='linear',tooltip='Integration timestep spacing. linear follows the reference grid; cosine concentrates steps near both endpoints. Output sigma = 1 - t.')],outputs=[io.Sigmas.Output(tooltip='CFM sigma grid decreasing from 1 to 0, for SamplerCustomAdvanced.')])

    @classmethod
    def execute(cls,steps,schedule='linear'):
        times=torch.linspace(0.,1.,steps+1,device=mm.get_torch_device())
        if schedule=='cosine':
            times=(1.-torch.cos(times*math.pi))*.5
        elif schedule!='linear':
            raise ValueError('Unsupported CFM schedule.')
        # Preserve each reference FP32 timestep exactly through sigma=1-t.
        return io.NodeOutput(1.-times.cpu().double())


class EmptyIndexTTSMelLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='EmptyIndexTTSMelLatent',display_name='Empty IndexTTS Mel Latent',category=CATEGORY+'/sampling',
            description='80-channel mel capacity at 22050 Hz / hop 256. Speech may end earlier at EOS.',
            inputs=[io.Float.Input('seconds',default=30.,min=.1,max=120.,step=.1,tooltip='Maximum mel capacity per text segment in seconds. EOS may end the output earlier.'),
                    io.Int.Input('batch_size',default=1,min=1,max=64,tooltip='Number of speech variants per text segment. Shorter variants are padded to the longest item in the batch.')],outputs=[io.Latent.Output(tooltip='80-channel mel capacity for SamplerCustomAdvanced.')])

    @classmethod
    def execute(cls,seconds,batch_size=1):
        frames=math.ceil(seconds*22050/256)
        return io.NodeOutput({'samples':torch.zeros(batch_size,80,frames,device=mm.intermediate_device()),'sample_rate':22050})


class IndexTTSJoinSegments(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='IndexTTSJoinSegments',display_name='IndexTTS Join Audio Segments',category=CATEGORY,
            description='Join the list produced by separately sampling and decoding each text segment.',
            is_input_list=True,inputs=[io.Audio.Input('audio',tooltip='Decoded audio segments in text order, from VAE Decode Audio.'),io.Float.Input('silence_ms',default=0.,min=0.,max=5000.,step=10.,tooltip='Silence inserted between adjacent segments, in milliseconds.')],
            outputs=[io.Audio.Output(tooltip='Audio segments joined into one waveform at the original sample rate.')])

    @classmethod
    def execute(cls,audio,silence_ms):
        rate=audio[0]['sample_rate']
        waves=[]
        for index,item in enumerate(audio):
            if item['sample_rate']!=rate:
                raise ValueError('All segments must have the same sample rate.')
            if index and silence_ms[0]>0:
                waves.append(torch.zeros((*item['waveform'].shape[:2],round(silence_ms[0]*rate/1000)),
                                         device=item['waveform'].device,dtype=item['waveform'].dtype))
            waves.append(item['waveform'])
        return io.NodeOutput({'waveform':torch.cat(waves,dim=-1),'sample_rate':rate})


CHECKPOINT_NODES=(IndexTTSCheckpointLoader,IndexTTSSpeakerConditioning,IndexTTSEmotionAudioConditioning,
    IndexTTSEmotionVectorConditioning,IndexTTSEmotionTextConditioning,IndexTTSGuiderNode,IndexTTSSamplerNode,
    IndexTTSRandomSourceNode,IndexTTSCFMSchedule,EmptyIndexTTSMelLatent,IndexTTSJoinSegments)
