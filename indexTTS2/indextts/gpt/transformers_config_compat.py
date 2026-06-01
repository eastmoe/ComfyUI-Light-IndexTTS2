# coding=utf-8
"""Compatibility helpers for Transformers config API changes."""


def get_non_default_generation_parameters(config):
    """Return generation params stored on a model config across Transformers versions."""
    get_params = getattr(config, "_get_non_default_generation_parameters", None)
    if callable(get_params):
        return get_params()

    get_params = getattr(config, "_get_generation_parameters", None)
    if callable(get_params):
        return get_params()

    return {}


def ensure_model_config_compat(config):
    """Add legacy PretrainedConfig attributes expected by vendored modeling code."""
    legacy_defaults = {
        "loss_type": None,
        "pruned_heads": {},
        "sep_token_id": None,
        "torchscript": False,
    }

    for attr_name, default_value in legacy_defaults.items():
        if not hasattr(config, attr_name):
            setattr(config, attr_name, default_value.copy() if isinstance(default_value, dict) else default_value)
    return config


def ensure_generation_config_compat(generation_config, model_kwargs=None):
    """Add legacy GenerationConfig attributes expected by vendored generation code."""
    legacy_defaults = {
        "bad_words_ids": None,
        "begin_suppress_tokens": None,
        "constraints": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "encoder_repetition_penalty": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
        "exponential_decay_length_penalty": None,
        "force_words_ids": None,
        "forced_bos_token_id": None,
        "forced_decoder_ids": None,
        "forced_eos_token_id": None,
        "guidance_scale": None,
        "length_penalty": 1.0,
        "low_memory": False,
        "max_time": None,
        "min_length": 0,
        "min_new_tokens": None,
        "min_p": None,
        "no_repeat_ngram_size": 0,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_return_sequences": 1,
        "output_logits": False,
        "output_scores": False,
        "penalty_alpha": None,
        "remove_invalid_values": False,
        "renormalize_logits": False,
        "return_legacy_cache": None,
        "sequence_bias": None,
        "stop_strings": None,
        "suppress_tokens": None,
        "temperature": 1.0,
        "token_healing": False,
        "top_k": 50,
        "top_p": 1.0,
        "typical_p": 1.0,
        "watermarking_config": None,
    }

    for attr_name, default_value in legacy_defaults.items():
        if model_kwargs is not None and attr_name in model_kwargs:
            value = model_kwargs.pop(attr_name)
            setattr(generation_config, attr_name, default_value if value is None else value)
        elif not hasattr(generation_config, attr_name) or getattr(generation_config, attr_name) is None:
            setattr(generation_config, attr_name, default_value)

    return generation_config
