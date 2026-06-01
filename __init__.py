from __future__ import annotations

try:
    from .src.comfy_light_indextts2.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .src.comfy_light_indextts2.paths import register_model_folder

    register_model_folder()
except Exception as exc:
    print(f"[Comfy-Light-IndexTTS2] Failed to import nodes: {exc}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./static"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
