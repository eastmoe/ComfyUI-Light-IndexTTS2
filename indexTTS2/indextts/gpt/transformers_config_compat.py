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


def ensure_generation_config_compat(generation_config, model_kwargs=None):
    """Add legacy GenerationConfig attributes expected by vendored generation code."""
    legacy_defaults = {
        "forced_decoder_ids": None,
        "return_legacy_cache": None,
    }

    for attr_name, default_value in legacy_defaults.items():
        if model_kwargs is not None and attr_name in model_kwargs:
            setattr(generation_config, attr_name, model_kwargs.pop(attr_name))
        elif not hasattr(generation_config, attr_name):
            setattr(generation_config, attr_name, default_value)

    return generation_config
