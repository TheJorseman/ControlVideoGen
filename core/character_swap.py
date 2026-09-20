"""Swap de persona con Nano Banana (API de imagenes de Gemini).

Dado un frame del video de referencia y una foto de la persona nueva,
genera una imagen de la persona nueva en la pose/escena del frame,
que luego sirve como 'character image' para Wan Animate.
"""
from __future__ import annotations

import base64
from pathlib import Path

NANO_MODELS = {
    "Nano Banana (rapido)": "gemini-2.5-flash-image",
    "Nano Banana Pro (mejor)": "gemini-3-pro-image-preview",
}

_EDIT_PROMPT = (
    "Image 1 is a frame from a video. Image 2 is the new person. "
    "Replace ONLY the main person's identity (face, hair, body, clothing style) in "
    "Image 1 with the person from Image 2, keeping the exact same pose, framing, "
    "lighting and background of Image 1. Photorealistic, seamless blend, no artifacts."
)


def _to_b64(path: str, max_side: int = 1024) -> tuple[str, str]:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    if max(im.size) > max_side:
        scale = max_side / max(im.size)
        im = im.resize((int(im.width * scale), int(im.height * scale)))
    import io

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), "image/png"


def generate_character_reference(video_path: str, person_image_path: str,
                                 api_key: str, model: str = "gemini-2.5-flash-image",
                                 frame_pick: float = 0.3) -> str:
    """frame_pick: posicion relativa (0-1) del frame a usar como escena base."""
    import httpx

    from . import preprocess
    from .config import OUTPUT_DIR

    if not api_key:
        raise RuntimeError("Falta la API key de Gemini en Modelos y Ajustes.")

    frames, _fps = preprocess.load_video(video_path, max_frames=120)
    frame = frames[min(int(len(frames) * frame_pick), len(frames) - 1)]
    import PIL.Image

    tmp = OUTPUT_DIR / "_scene_frame.png"
    PIL.Image.fromarray(frame).save(tmp)

    scene_b64, scene_mime = _to_b64(str(tmp))
    person_b64, person_mime = _to_b64(person_image_path)

    resp = httpx.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json={
            "contents": [{
                "parts": [
                    {"text": _EDIT_PROMPT},
                    {"inline_data": {"mime_type": scene_mime, "data": scene_b64}},
                    {"inline_data": {"mime_type": person_mime, "data": person_b64}},
                ]
            }]
        },
        timeout=120,
    )
    resp.raise_for_status()
    parts = resp.json()["candidates"][0]["content"]["parts"]
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            out = OUTPUT_DIR / f"character_ref_{Path(video_path).stem}.png"
            out.write_bytes(base64.b64decode(inline["data"]))
            return str(out)
    raise RuntimeError("La API no devolvio una imagen (revisa el prompt/seguridad).")
