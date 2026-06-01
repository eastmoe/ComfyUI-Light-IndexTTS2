from __future__ import annotations

from pathlib import Path

try:
    import folder_paths
except ImportError:
    folder_paths = None


MODEL_FOLDER_NAME = "IndexTTS2"
DEFAULT_MODEL_SUBDIR = "IndexTTS-2"


def comfy_models_dir() -> Path:
    if folder_paths is not None and getattr(folder_paths, "models_dir", None):
        return Path(folder_paths.models_dir)
    return Path.cwd() / "models"


def comfy_output_dir() -> Path:
    if folder_paths is not None and hasattr(folder_paths, "get_output_directory"):
        return Path(folder_paths.get_output_directory())
    return Path.cwd() / "output"


def comfy_temp_dir() -> Path:
    if folder_paths is not None and hasattr(folder_paths, "get_temp_directory"):
        return Path(folder_paths.get_temp_directory())
    return Path.cwd() / "temp"


def default_model_dir() -> Path:
    return default_model_root() / DEFAULT_MODEL_SUBDIR


def default_model_root() -> Path:
    return comfy_models_dir() / MODEL_FOLDER_NAME


def resolve_path(value: str | None, default: Path | None = None) -> Path:
    text = (value or "").strip()
    if not text or text.lower() == "auto":
        if default is None:
            raise ValueError("Path is set to auto, but no default was provided.")
        return default
    return Path(text).expanduser()


def resolve_output_path(output_path: str) -> Path | None:
    text = (output_path or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_absolute():
        return path
    return comfy_output_dir() / path


def register_model_folder() -> None:
    if folder_paths is None or not hasattr(folder_paths, "folder_names_and_paths"):
        return
    model_dir = default_model_root()
    current = folder_paths.folder_names_and_paths.get(MODEL_FOLDER_NAME)
    if current is None:
        folder_paths.folder_names_and_paths[MODEL_FOLDER_NAME] = ([str(model_dir)], set())
