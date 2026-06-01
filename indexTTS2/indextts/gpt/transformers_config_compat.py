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
