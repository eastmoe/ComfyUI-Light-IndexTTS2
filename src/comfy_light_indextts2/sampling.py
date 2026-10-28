from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import comfy.model_management as mm
import comfy.utils

from indextts.gpt.beam_search import BeamSearchScorer
from indextts.utils.typical_sampling import TypicalLogitsWarper
from .backend import IndexTTSModel


class IndexTTSRandomSource:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, latent):
        return torch.zeros_like(latent['samples'])


@dataclass
class IndexTTSSampler:
    max_tokens: int = 1500
    do_sample: bool = True
    temperature: float = 0.8
    top_k: int = 30
    top_p: float = 0.8
    repetition_penalty: float = 10.
    num_beams: int = 1
    length_penalty: float = 0.
    typical_mass: float = 1.
    cfm_cfg: float = 0.7
    cfm_temperature: float = 1.
    length_scale: float = 1.72
    target_duration: float = 0.

    def process_logits(self, ids, scores):
        repeated = scores.gather(1, ids)
        repeated = torch.where(repeated < 0, repeated * self.repetition_penalty, repeated / self.repetition_penalty)
        scores = scores.scatter(1, ids, repeated)
        keep = 2 if self.num_beams > 1 else 1
        if self.typical_mass < 1.:
            scores = TypicalLogitsWarper(mass=self.typical_mass, min_tokens_to_keep=keep)(ids, scores)
        if self.do_sample:
            scores = scores / self.temperature
            if self.top_k:
                count = min(max(self.top_k, keep), scores.shape[-1])
                scores = scores.masked_fill(scores < scores.topk(count).values[:, -1:], -torch.inf)
            if self.top_p < 1.:
                sorted_scores, indices = scores.sort(descending=False)
                remove = sorted_scores.softmax(-1).cumsum(-1) <= (1 - self.top_p)
                remove[..., -keep:] = False
                scores = scores.masked_fill(remove.scatter(1, indices, remove), -torch.inf)
        return scores

    def generate_codes(self, model, text, prefix, generator, limit, progress):
        gpt = model.gpt
        ids, embedded, _ = gpt.prepare_gpt_inputs(prefix, text)
        prompt_length = ids.shape[1]
        initial = gpt.mel_embedding(ids[:,-1:]) + gpt.mel_pos_embedding.emb.weight[0]
        inputs = torch.cat((embedded, initial),dim=1)
        beams = self.num_beams
        scorer = None
        if beams > 1:
            scorer = BeamSearchScorer(batch_size=1,num_beams=beams,device=text.device,
                length_penalty=self.length_penalty,do_early_stopping=False,max_length=prompt_length+limit)
            ids, inputs = ids.repeat(beams,1), inputs.repeat(beams,1,1)
            beam_scores = torch.zeros(beams,device=text.device)
            beam_scores[1:] = -1e9
        cache = None
        for step in range(limit):
            mm.throw_exception_if_processing_interrupted()
            output = gpt.gpt(inputs_embeds=inputs,past_key_values=cache,use_cache=True)
            cache = output.past_key_values
            logits = gpt.mel_head(gpt.final_norm(output.last_hidden_state[:,-1:]))[:,0].float()
            scores = F.log_softmax(logits,dim=-1) if beams>1 else logits
            scores = self.process_logits(ids, scores)
            if beams > 1:
                combined = (scores + beam_scores[:,None]).reshape(1,-1)
                if self.do_sample:
                    choices = torch.multinomial(combined.softmax(-1).to(generator.device),2*beams,generator=generator).to(text.device)
                    values,order = combined.gather(1,choices).sort(dim=1,descending=True)
                    choices = choices.gather(1,order)
                else:
                    values,choices = combined.topk(2*beams,dim=1)
                next_indices,next_tokens = choices//scores.shape[-1],choices%scores.shape[-1]
                selected = scorer.process(ids,values,next_tokens,next_indices,pad_token_id=8193,eos_token_id=8193,
                                           decoder_prompt_len=prompt_length)
                beam_scores,token,indices = (selected[k] for k in ('next_beam_scores','next_beam_tokens','next_beam_indices'))
                ids = torch.cat((ids[indices],token[:,None]),dim=1)
                cache = tuple(tuple(value[indices] for value in layer) for layer in cache)
            else:
                token = (torch.multinomial(scores.softmax(-1).to(generator.device),1,generator=generator).to(text.device)
                         if self.do_sample else scores.argmax(-1,keepdim=True)).reshape(-1)
                ids = torch.cat((ids,token[:,None]),dim=1)
            progress(step+1)
            if (scorer is not None and scorer.is_done) or (beams==1 and token.item()==8193):
                break
            # BOS is position 0; the first generated token is incremental input 1.
            inputs = gpt.mel_embedding(token[:,None]) + gpt.mel_pos_embedding.emb.weight[step+1]
        if scorer is not None:
            ids = scorer.finalize(ids,beam_scores,next_tokens,next_indices,max_length=prompt_length+limit,
                                   pad_token_id=8193,eos_token_id=8193,decoder_prompt_len=prompt_length)['sequences']
        codes = ids[:1,prompt_length:]
        eos = (codes[0]==8193).nonzero()
        if eos.numel():
            codes = codes[:,:eos[0,0].item()]
        if not codes.shape[1]:
            raise ValueError('IndexTTS produced no semantic tokens. Try another reference or seed.')
        return codes

    def integrate(self, estimator, condition, reference, style, sigmas, generator, progress):
        batch,total,_ = condition.shape
        length = reference.shape[-1]
        x = torch.randn((batch,80,total),device=generator.device,generator=generator).to(condition.device)*self.cfm_temperature
        x[...,:length] = 0
        prompt = torch.zeros_like(x)
        prompt[...,:length] = reference
        times = (1.-sigmas.cpu().double()).float().to(x.device)
        time = times[0]
        lengths = torch.full((batch,),total,device=x.device,dtype=torch.long)
        for step in range(1,len(times)):
            mm.throw_exception_if_processing_interrupted()
            dt = times[step]-times[step-1]
            if self.cfm_cfg > 0:
                velocity = estimator(torch.cat((x,x)),torch.cat((prompt,torch.zeros_like(prompt))),lengths,
                    torch.cat((time[None],time[None])),torch.cat((style,torch.zeros_like(style))),
                    torch.cat((condition,torch.zeros_like(condition))))
                positive,negative = velocity.chunk(2,dim=0)
                velocity = (1.+self.cfm_cfg)*positive-self.cfm_cfg*negative
            else:
                velocity = estimator(x,prompt,lengths,time[None],style,condition)
            x = x+dt*velocity
            time = time+dt
            x[...,:length] = 0
            progress(step)
        return x[...,length:]


class IndexTTSGuider:
    def __init__(self, model, conditioning):
        if not isinstance(model.model,IndexTTSModel):
            raise ValueError('Connect MODEL from an IndexTTS checkpoint.')
        self.model_patcher,self.conditioning = model,conditioning

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None,
               disable_pbar=False, seed=None):
        if not isinstance(sampler,IndexTTSSampler):
            raise TypeError('Connect the IndexTTS AR / CFM Sampler.')
        if sigmas.numel()<2 or not torch.isfinite(sigmas).all():
            raise ValueError('CFM sigmas must contain at least two finite values.')
        patcher = self.model_patcher
        model = patcher.model
        text,metadata = self.conditioning[0]
        data = metadata['indextts']
        if data['checkpoint_id'] != model.checkpoint_id:
            raise ValueError('Use MODEL and conditioning from the same IndexTTS checkpoint.')
        device,dtype = patcher.load_device,model.compute_dtype
        batch_size = latent_image.shape[0]
        limit = min(sampler.max_tokens,max(1,int(latent_image.shape[-1]/sampler.length_scale)),model.config.gpt.max_mel_tokens)
        cache_bytes = model.config.gpt.layers*2*model.config.gpt.model_dim*(text.shape[-1]+36+limit)*sampler.num_beams*mm.dtype_size(dtype)
        mm.load_models_gpu([patcher],memory_required=1024*1024**2+cache_bytes,force_full_load=True)
        total = batch_size*(limit+len(sigmas)-1)
        progress = comfy.utils.ProgressBar(total) if callback is None else None
        results=[]
        patcher.pre_run()
        try:
            speaker,emotion,style,reference = (data[key].to(device) for key in ('speaker','emotion','style','ref_mel'))
            text = text.to(device)
            lengths = torch.full((1,),speaker.shape[1],device=device,dtype=torch.long)
            emo_lengths = torch.full((1,),emotion.shape[1],device=device,dtype=torch.long)
            _,semantic_reference = model.semantic_codec.quantize(speaker)
            ref_lengths = torch.full((1,),reference.shape[-1],device=device,dtype=torch.long)
            prompt = model.s2mel['length_regulator'](semantic_reference,ylens=ref_lengths,n_quantizers=3,f0=None)[0]
            enabled = dtype in (torch.float16,torch.bfloat16)
            with torch.autocast(device.type,dtype=dtype,enabled=enabled):
                emotion_vector = model.gpt.merge_emovec(speaker,emotion,lengths,emo_lengths,alpha=data['emotion_alpha'])
                if 'emotion_vector' in data:
                    emotion_vector = data['emotion_vector'].to(device)+(1.-data['emotion_weight'])*emotion_vector
                voice = model.gpt.get_conditioning(speaker.transpose(1,2),lengths)
                if sampler.target_duration > 0:
                    count = int(sampler.target_duration*50)
                    duration = model.gpt.mel_pos_embedding.emb.weight[count:count+1]
                    half = model.gpt.mel_pos_embedding.emb.weight[count//2:count//2+1]
                else:
                    half,duration = model.gpt.speed_emb.weight[1:2],model.gpt.speed_emb.weight[0:1]
                prefix = torch.cat((voice+emotion_vector[:,None],half[:,None],duration[:,None]),dim=1)
                with tqdm(total=total,disable=disable_pbar) as bar:
                    for batch in range(batch_size):
                        generator_device = device if device.type in ('cuda','cpu','xpu') else torch.device('cpu')
                        generator = torch.Generator(device=generator_device)
                        generator.manual_seed(((seed or 0)+data['segment_index']*batch_size+batch)%(2**64))
                        offset = batch*(limit+len(sigmas)-1)
                        completed = 0
                        def update(value):
                            nonlocal completed
                            bar.update(value-completed)
                            completed=value
                            if callback is not None:
                                callback(offset+value-1,latent_image,latent_image,total)
                            else:
                                progress.update_absolute(offset+value)
                        codes = sampler.generate_codes(model,text,prefix,generator,limit,update)
                        update(limit)
                        code_lengths = torch.full((1,),codes.shape[1],device=device,dtype=torch.long)
                        use_speed = torch.full((1,),int(sampler.target_duration>0),device=device,dtype=torch.long)
                        latent = model.gpt(speaker,text.clone(),torch.full((1,),text.shape[1],device=device,dtype=torch.long),
                            codes.clone(),code_lengths,emotion,cond_mel_lengths=lengths,emo_cond_mel_lengths=emo_lengths,
                            emo_vec=emotion_vector,use_speed=use_speed)
                        semantic = model.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1)).transpose(1,2)
                        semantic = semantic+model.s2mel['gpt_layer'](latent)
                        target = torch.full((1,),int(codes.shape[1]*sampler.length_scale),device=device,dtype=torch.long)
                        cond = model.s2mel['length_regulator'](semantic,ylens=target,n_quantizers=3,f0=None)[0]
                        mel = sampler.integrate(model.s2mel['cfm'].estimator,torch.cat((prompt,cond),dim=1),reference,
                                                 style,sigmas,generator,lambda i:update(limit+i))
                        results.append(mel.float())
            frames=max(item.shape[-1] for item in results)
            output=torch.full((batch_size,80,frames),-11.512925,device=device)
            for batch,item in enumerate(results):
                output[batch,:,:item.shape[-1]]=item[0]
            if callback is not None:
                callback(total-1,output,output,total)
            return output
        finally:
            patcher.cleanup()
