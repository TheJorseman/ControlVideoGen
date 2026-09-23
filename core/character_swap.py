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


def gemini_request(model: str, api_key: str, payload: dict, timeout: int = 120,
                   retries: int = 3):
    """POST a Gemini generateContent con auth por header (la key nunca va en la URL),
    reintentos con backoff para 429/503 y errores legibles."""
    import time

    import httpx

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    last_status = None
    for attempt in range(retries):
        try:
            r = httpx.post(url, headers={"x-goog-api-key": api_key},
                           json=payload, timeout=timeout)
        except httpx.TimeoutException:
            if attempt == retries - 1:
                raise
            time.sleep(3)
            continue
        if r.status_code in (429, 503) and attempt < retries - 1:
            wait = 0
            try:
                det = r.json().get("error", {}).get("details", [])
                for d in det:
                    rd = d.get("retryDelay", "")
                    if rd.endswith("s"):
                        wait = int(float(rd[:-1]))
                        break
            except Exception:  # noqa: BLE001
                pass
            wait = wait or int(r.headers.get("retry-after") or (10 * (attempt + 1)))
            time.sleep(min(wait, 60))
            continue
        if r.status_code >= 400:
            detail = r.text[:300]
            try:
                err = r.json().get("error", {})
                detail = err.get("message") or detail
                status = err.get("status", "")
                if status == "RESOURCE_EXHAUSTED" or r.status_code == 429:
                    detail += (" — Cuota/RPM agotada: espera ~1 min entre peticiones. "
                               "Las imagenes de Gemini en nivel gratuito tienen limite "
                               "muy bajo (o requieren facturacion activa).")
                if "API_KEY_INVALID" in status or r.status_code == 401:
                    detail += " — Revisa/renueva la API key de Gemini en Ajustes."
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(f"Gemini {r.status_code}: {detail}")
        return r
    raise RuntimeError("Gemini: reintentos agotados (limit rate sostenido)")


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


SWAP_ENGINES = {
    "Nano Banana (Gemini)": "gemini",
    "GPT-Image-1 (OpenAI)": "openai",
    "MiniMax image-01": "minimax",
}


def _scene_frame(video_path: str, frame_pick: float) -> str:
    from . import preprocess
    from .config import OUTPUT_DIR

    frames, _fps = preprocess.load_video(video_path, max_frames=120)
    frame = frames[min(int(len(frames) * frame_pick), len(frames) - 1)]
    import PIL.Image

    tmp = OUTPUT_DIR / "_scene_frame.png"
    PIL.Image.fromarray(frame).save(tmp)
    return str(tmp)


def swap_with_openai(video_path: str, person_image_path: str, api_key: str,
                     model: str = "gpt-image-1", frame_pick: float = 0.3) -> str:
    """Alternativa al Nano Banana: edicion de imagen via OpenAI /v1/images/edits."""
    import base64

    import httpx

    from .config import OUTPUT_DIR

    if not api_key:
        raise RuntimeError("Falta la API key de OpenAI en Modelos y Ajustes.")

    scene = _scene_frame(video_path, frame_pick)
    with open(scene, "rb") as fa, open(person_image_path, "rb") as fp:
        r = httpx.post(
            "https://api.openai.com/v1/images/edits",
            headers={"authorization": f"Bearer {api_key}"},
            files=[
                ("image[]", ("scene.png", fa, "image/png")),
                ("image[]", ("person.png", fp, "image/png")),
            ],
            data={"model": model, "prompt": _EDIT_PROMPT, "size": "1024x1024"},
            timeout=240,
        )
    if r.status_code >= 400:
        detail = r.text[:300]
        try:
            detail = r.json().get("error", {}).get("message", detail)
        except Exception:  # noqa: BLE001
            pass
        if r.status_code in (401, 403):
            detail += " — Revisa la API key de OpenAI (Ajustes)."
        if r.status_code == 429:
            detail += " — Cuota/RPM de OpenAI agotada: espera o revisa facturacion."
        raise RuntimeError(f"OpenAI {r.status_code}: {detail}")
    data = r.json()["data"][0]
    out = OUTPUT_DIR / f"character_ref_{Path(video_path).stem}_openai.png"
    if data.get("b64_json"):
        out.write_bytes(base64.b64decode(data["b64_json"]))
    elif data.get("url"):
        out.write_bytes(httpx.get(data["url"], timeout=120).content)
    else:
        raise RuntimeError("OpenAI no devolvio imagen.")
    return str(out)


def swap_with_minimax(video_path: str, person_image_path: str, api_key: str,
                      scene_prompt: str = "", frame_pick: float = 0.3,
                      host: str = "") -> str:
    """image-01 con subject_reference: retrata a la persona nueva (identidad
    consistente). scene_prompt describe el contexto; si no hay, foto limpia."""
    from .config import OUTPUT_DIR

    if not api_key:
        raise RuntimeError("Falta la API key de MiniMax en Modelos y Ajustes.")

    _scene_frame(video_path, frame_pick)  # valida que el video se puede leer
    prompt = (scene_prompt.strip() + ", " if scene_prompt.strip() else "") + (
        "photorealistic full-body portrait of the referenced person, natural pose, "
        "clear face, sharp details"
    )
    from .backends import minimax_api

    data = minimax_api.generate_image(api_key, prompt,
                                      reference_path=person_image_path,
                                      aspect_ratio="1:1", host=host)
    out = OUTPUT_DIR / f"character_ref_{Path(video_path).stem}_minimax.png"
    out.write_bytes(data)
    return str(out)


def generate_character_reference(video_path: str, person_image_path: str,
                                 api_key: str, model: str = "gemini-2.5-flash-image",
                                 frame_pick: float = 0.3) -> str:
    """frame_pick: posicion relativa (0-1) del frame a usar como escena base."""
    from .config import OUTPUT_DIR

    if not api_key:
        raise RuntimeError("Falta la API key de Gemini en Modelos y Ajustes.")

    scene = _scene_frame(video_path, frame_pick)
    scene_b64, scene_mime = _to_b64(scene)
    person_b64, person_mime = _to_b64(person_image_path)

    payload = {
        "contents": [{
            "parts": [
                {"text": _EDIT_PROMPT},
                {"inline_data": {"mime_type": scene_mime, "data": scene_b64}},
                {"inline_data": {"mime_type": person_mime, "data": person_b64}},
            ]
        }]
    }
    resp = gemini_request(model, api_key, payload, timeout=120)
    parts = resp.json()["candidates"][0]["content"]["parts"]
    for part in parts:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            out = OUTPUT_DIR / f"character_ref_{Path(video_path).stem}.png"
            out.write_bytes(base64.b64decode(inline["data"]))
            return str(out)
    raise RuntimeError("La API no devolvio una imagen (revisa el prompt/seguridad).")
