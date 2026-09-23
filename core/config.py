"""Configuracion persistente de la app (settings.json).

apply_env() debe llamarse ANTES de importar torch/diffusers/huggingface_hub
para que todas las descargas vayan al directorio de modelos elegido.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
SETTINGS_PATH = APP_DIR / "settings.json"
OUTPUT_DIR = APP_DIR / "outputs"

DEFAULT_MODEL_DIR = str(APP_DIR / "models")

DEFAULTS: dict = {
    "model_dir": DEFAULT_MODEL_DIR,
    "vram_profile": "auto",
    "hf_token": "",
    "minimax_host": "https://api.minimax.io",
    "api_keys": {
        "deepseek": "",
        "openai": "",
        "anthropic": "",
        "gemini": "",
        "minimax": "",
    },
}


def load_settings() -> dict:
    settings = json.loads(json.dumps(DEFAULTS))
    if SETTINGS_PATH.exists():
        try:
            saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))
            for key, value in saved.items():
                if key == "api_keys" and isinstance(value, dict):
                    settings["api_keys"].update(value)
                else:
                    settings[key] = value
        except (json.JSONDecodeError, OSError):
            pass
    return settings


def save_settings(settings: dict) -> None:
    SETTINGS_PATH.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def model_path(model_dir: str, folder: str) -> Path:
    """Ruta local de un modelo dentro del gestor de modelos."""
    return Path(model_dir) / folder


def apply_env(model_dir: str) -> None:
    """Redirige todas las cachés de HuggingFace al directorio de modelos."""
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(model_dir)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path(model_dir) / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(Path(model_dir) / "hub")
    os.environ["U2NET_HOME"] = str(Path(model_dir) / "preproc")
    os.environ["XDG_CACHE_HOME"] = str(Path(model_dir) / "preproc")
