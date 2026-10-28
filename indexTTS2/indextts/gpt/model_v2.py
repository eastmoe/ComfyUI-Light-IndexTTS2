import torch
import torch.nn as nn
import torch.nn.functional as F

from indextts.gpt.gpt2 import GPT

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))


class UnifiedVoice(nn.Module):
    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1,
                 mel_length_compression=1024, number_text_tokens=256,
                 start_text_token=0, stop_text_token=1, number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193,
                 train_solo_embeddings=False, use_mel_codes_as_input=True,
                 types=1,
                 condition_num_latent=32, condition_type="conformer_perceiver", condition_module=None, emo_condition_module=None):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            condition_type: perceiver, gst or default encoder
        """
        super().__init__()
        if condition_type != 'conformer_perceiver' or not use_mel_codes_as_input or train_solo_embeddings:
            raise ValueError('Unsupported IndexTTS checkpoint architecture.')
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.cond_num = condition_num_latent
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        self.conditioning_encoder = ConformerEncoder(
            input_size=1024, output_size=condition_module['output_size'],
            linear_units=condition_module['linear_units'], attention_heads=condition_module['attention_heads'],
            num_blocks=condition_module['num_blocks'], input_layer=condition_module['input_layer'])
        self.perceiver_encoder = PerceiverResampler(
            model_dim, dim_context=condition_module['output_size'], ff_mult=condition_module['perceiver_mult'],
            heads=condition_module['attention_heads'], num_latents=self.cond_num)

        self.emo_conditioning_encoder = ConformerEncoder(input_size=1024,
                                                         output_size=emo_condition_module['output_size'],
                                                         linear_units=emo_condition_module['linear_units'],
                                                         attention_heads=emo_condition_module['attention_heads'],
                                                         num_blocks=emo_condition_module['num_blocks'],
                                                         input_layer=emo_condition_module['input_layer'])
        self.emo_perceiver_encoder = PerceiverResampler(1024, dim_context=emo_condition_module['output_size'],
                                                            ff_mult=emo_condition_module['perceiver_mult'],
                                                            heads=emo_condition_module['attention_heads'],
                                                            num_latents=1)


        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)

        self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        self.gpt = GPT(layers, model_dim, heads)
        self.mel_pos_embedding = LearnedPositionEmbeddings(self.max_mel_tokens + 2 + self.max_conditioning_inputs, model_dim)
        self.text_pos_embedding = LearnedPositionEmbeddings(self.max_text_tokens + 2, model_dim)
        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        self.speed_emb = nn.Embedding(2, model_dim)

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_logits(self, speech_conditioning_inputs, first_inputs, first_head, second_inputs=None, second_head=None, get_attns=False, return_latent=False):
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns)
        if get_attns:
            return gpt_out.attentions

        offset = speech_conditioning_inputs.shape[1]
        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, :first_inputs.shape[1]], enc[:, -second_inputs.shape[1]:]

        first_logits = enc[:, :first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1]:]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        features, mask = self.conditioning_encoder(speech_conditioning_input.transpose(1, 2), cond_mel_lengths)
        return self.perceiver_encoder(features, self.cond_mask_pad(mask.squeeze(1)))

    def get_emo_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        speech_conditioning_input, mask = self.emo_conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                        cond_mel_lengths)  # (b, s, d), (b, 1, s)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 1, d)
        return conds.squeeze(1)


    def forward(self, speech_conditioning_latent, text_inputs, text_lengths, mel_codes, mel_codes_lengths, emo_speech_conditioning_latent,
                cond_mel_lengths=None, emo_cond_mel_lengths=None, emo_vec=None, use_speed=None):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode

        speech_conditioning_input: MEL float tensor, (b,1024)
        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)

        If return_attentions is specified, only logits are returned.
        If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """

        speech_conditioning_latent = self.get_conditioning(speech_conditioning_latent.transpose(1,2), cond_mel_lengths)

        if emo_vec is None:
            emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_mel_lengths)
            emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
            emo_vec = self.emo_layer(emo_vec_syn)

        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)

        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        duration_emb = self.mel_pos_embedding.emb(mel_codes_lengths) * use_speed.unsqueeze(1)
        duration_emb_half = self.mel_pos_embedding.emb(mel_codes_lengths // 2) * use_speed.unsqueeze(1)

        speed_emb = self.speed_emb(torch.zeros_like(use_speed)) * (1 - use_speed).unsqueeze(1)
        duration_emb = duration_emb + speed_emb
        speed_emb_half = self.speed_emb(torch.ones_like(use_speed)) * (1 - use_speed).unsqueeze(1)
        duration_emb_half = duration_emb_half + speed_emb_half

        conds = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)

        mel_emb = self.mel_embedding(mel_codes)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        text_logits, mel_logits = self.get_logits(conds, text_emb, self.text_head, mel_emb, self.mel_head, get_attns=False, return_latent=True)
        return mel_logits[:, :-2]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs: torch.Tensor,
    ):
        
        """
        Prepare the text prefix and BOS token for autoregressive sampling.
        Args:
            conds_latent: (b, 32, dim) audio conditioning embedding by `get_conditioning()`
            text_inputs: (b, L)
        Returns:
            input_ids: (b, s+1) prefix placeholders followed by the mel BOS token
            inputs_embeds: (b, s, dim) condition and text embeddings
            attention_mask: (b, s+1) valid prefix positions and mel BOS
        """
        b, L = text_inputs.shape[:2]
        device = text_inputs.device
        single_cond = conditional_latents.ndim == 3 and conditional_latents.shape[0] == 1
        if not single_cond:
            assert conditional_latents.shape[0] == b, f"batch size mismatch: {conditional_latents.shape[0]} vs {b}"
        batched_mel_emb = []
        attention_masks = []
        target_len = conditional_latents.shape[1] + L + 2
        for i in range(b):
            valid_mask = (text_inputs[i] != self.stop_text_token) & (text_inputs[i] != self.start_text_token)
            text_input = text_inputs[i][valid_mask]
            text_input = F.pad(text_input, (1, 0), value=self.start_text_token)
            text_input = F.pad(text_input, (0, 1), value=self.stop_text_token)
            text_input_pos = torch.arange(0, text_input.size(-1), device=device)
            text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)
            # concatenate [conditional latents][text embeddings]
            conds_text_emb = [
                conditional_latents.squeeze(0) if single_cond else conditional_latents[i],
                text_emb,
            ]
            # +1 for the start_mel_token
            attention_mask = torch.ones(target_len+1, dtype=torch.long, device=device)
            # check this text input is padded
            padding: int = L + 2 - text_input.size(-1)
            # pad left of [cond][text] -> [pad][cond][text]
            if padding > 0:
                pad = torch.zeros((padding, conditional_latents.size(-1)), dtype=text_emb.dtype, device=device) # [p, dim]
                conds_text_emb.insert(0, pad)
                attention_mask[:padding] = 0
            mel_emb = torch.cat(conds_text_emb) #[s, dim]
            assert mel_emb.shape[0] == target_len, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"
            batched_mel_emb.append(mel_emb)
            attention_masks.append(attention_mask)
        # [b, s, dim]
        batched_mel_emb = torch.stack(batched_mel_emb, dim=0)
        # [b, s+1]
        attention_mask = torch.stack(attention_masks, dim=0)
        # [b, s+1]
        fake_inputs = torch.ones(
            (
                batched_mel_emb.shape[0],
                batched_mel_emb.shape[1] + 1,  # +1 for the start_mel_token
            ),
            dtype=torch.long,
            device=device,
        )
        fake_inputs[:, -1] = self.start_mel_token
        return fake_inputs, batched_mel_emb, attention_mask


    def get_emovec(self, emo_speech_conditioning_latent, emo_cond_lengths):
        emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        emo_vec = self.emo_layer(emo_vec_syn)
        return emo_vec

    def merge_emovec(self, speech_conditioning_latent, emo_speech_conditioning_latent, cond_lengths, emo_cond_lengths, alpha = 1.0):
        emo_vec = self.get_emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)

        out = base_vec + alpha * (emo_vec - base_vec)
        return out
