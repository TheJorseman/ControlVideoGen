"""Backend MiniMax H3 via API oficial (POST /v2/video_generation + polling).

Requisitos: API key de MiniMax (plan pay-as-you-go) en Ajustes.
"""
from __future__ import annotations

BASE = "https://api.minimax.io"
MODEL = "MiniMax-H3"
IMAGE_MODEL = "image-01"


def _base(host: str = "") -> str:
    return (host or BASE).rstrip("/")

_ERR_MSG = {
    1002: "Rate limit alcanzado: reintenta en unos segundos.",
    1004: "Autenticacion fallida: revisa la API key de MiniMax.",
    1008: "Saldo insuficiente en tu cuenta MiniMax (recarga creditos).",
    1026: "Contenido sensible detectado en el prompt/imagenes.",
    2013: "Parametros invalidos.",
    2049: "API key invalida: regenerala en platform.minimax.io.",
    2056: "Limite del Token Plan alcanzado para video: necesitas subir de plan o "
          "comprar Creditos (el plan de tokens cubre chat/imagen/speech, video con cupo propio).",
}


def _headers(api_key: str) -> dict:
    return {"authorization": f"Bearer {api_key}"}


def _check(body: dict, what: str):
    br = body.get("base_resp") or {}
    code = br.get("status_code")
    if code not in (0, None):
        hint = _ERR_MSG.get(code, br.get("status_msg", ""))
        raise RuntimeError(f"MiniMax {what} (code {code}): {hint}")


def _data_url(path: str) -> str:
    import base64
    from pathlib import Path

    ext = Path(path).suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def generate_image(api_key: str, prompt: str, reference_path: str | None = None,
                   aspect_ratio: str = "1:1", seed: int | None = None,
                   model: str = IMAGE_MODEL, host: str = "") -> bytes:
    """image-01: text-to-image o con subject_reference (identidad de personaje)."""
    import base64

    import httpx

    host = _base(host)
    if not api_key:
        raise RuntimeError("Falta la API key de MiniMax (Ajustes).")
    payload: dict = {"model": model, "prompt": prompt, "aspect_ratio": aspect_ratio,
                     "response_format": "base64", "n": 1}
    if reference_path:
        payload["subject_reference"] = [{"type": "character",
                                         "image_file": _data_url(reference_path)}]
    if seed is not None:
        payload["seed"] = int(seed)
    resp = httpx.post(f"{host}/v1/image_generation", headers=_headers(api_key),
                      json=payload, timeout=180)
    body = resp.json() if "json" in resp.headers.get("content-type", "") else {}
    _check(body, "image_generation")
    resp.raise_for_status()
    _check(body, "image_generation")
    urls = (body.get("data") or {}).get("image_base64") or []
    if not urls:
        raise RuntimeError(f"MiniMax image no devolvio datos: {str(body)[:200]}")
    return base64.b64decode(urls[0])


def upload_media(path: str, api_key: str, host: str = "") -> str:
    """Sube un archivo local y devuelve su URL de descarga (API de ficheros v1)."""
    import httpx
    from pathlib import Path

    host = _base(host)
    with open(path, "rb") as fh:
        resp = httpx.post(
            f"{host}/v1/files/upload",
            headers=_headers(api_key),
            files={"file": (Path(path).name, fh)},
            data={"purpose": "video-generation"},
            timeout=300,
        )
    resp.raise_for_status()
    body = resp.json()
    _check(body, "files/upload")
    file_id = (body.get("file") or {}).get("file_id") or body.get("file_id")
    if not file_id:
        raise RuntimeError(f"No se recibio file_id: {str(body)[:200]}")
    resp = httpx.get(f"{host}/v1/files/retrieve", headers=_headers(api_key),
                     params={"file_id": file_id}, timeout=60)
    resp.raise_for_status()
    url = (resp.json().get("file") or {}).get("download_url")
    if not url:
        raise RuntimeError(f"MiniMax retrieve sin download_url: {str(resp.json())[:200]}")
    return url


def _build_content(prompt: str, image_path: str | None, last_image_path: str | None,
                   api_key: str, host: str = "") -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image_path:
        url = upload_media(image_path, api_key, host)
        content.append({"type": "image_url", "image_url": {"url": url},
                        "role": "first_frame"})
    if last_image_path:
        url = upload_media(last_image_path, api_key, host)
        content.append({"type": "image_url", "image_url": {"url": url},
                        "role": "last_frame"})
    return content


def _submit_and_wait(api_key: str, payload: dict, poll_cb=None, host: str = "") -> str:
    import time

    import httpx

    host = _base(host)
    resp = httpx.post(f"{host}/v2/video_generation", headers=_headers(api_key),
                      json=payload, timeout=120)
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        resp.raise_for_status()
        raise RuntimeError(f"MiniMax respuesta no-JSON (HTTP {resp.status_code})")
    if resp.status_code >= 400 and "base_resp" not in body:
        err = (body.get("error") or {}).get("message") or str(body)[:250]
        raise RuntimeError(f"MiniMax video_generation: {err}")
    _check(body, "video_generation")
    task_id = body.get("task_id") or body.get("task", {}).get("id")
    if not task_id:
        raise RuntimeError(f"MiniMax no devolvio task_id: {str(body)[:200]}")
    while True:
        time.sleep(10)
        if poll_cb:
            poll_cb("waiting")
        r = httpx.get(f"{host}/v2/query/video_generation/{task_id}",
                      headers=_headers(api_key), timeout=60)
        r.raise_for_status()
        task = r.json()["task"]
        status = task["status"]
        if status == "succeeded":
            return task["content"]["url"]
        if status in ("failed", "cancelled"):
            raise RuntimeError(f"Tarea {status}: {task.get('error')}")


def generate(api_key: str, prompt: str, kind: str = "t2v",
             image_path: str | None = None, last_image_path: str | None = None,
             duration: int = 5, resolution: str = "768P", ratio: str = "16:9",
             poll_cb=None, host: str = "") -> str:
    """Crea la tarea, hace polling y devuelve la URL del video final."""
    host = _base(host)
    if not api_key:
        raise RuntimeError("Falta la API key de MiniMax (Ajustes).")
    content = _build_content(prompt, image_path if kind == "i2v" else None,
                             last_image_path if kind == "i2v" else None, api_key, host)
    payload = {"model": MODEL, "content": content, "duration": int(duration),
               "resolution": resolution}
    if kind == "t2v":
        payload["ratio"] = ratio
    return _submit_and_wait(api_key, payload, poll_cb, host)


def ref2va_generate(api_key: str, prompt: str, person_image_paths: list[str],
                    video_path: str | None = None, duration: int = 5,
                    resolution: str = "768P", ratio: str = "adaptive",
                    poll_cb=None, host: str = "") -> str:
    """H3 ref2va en la nube: persona(s) de referencia actuando como el video de
    referencia. Sin GPU local: el swap completo ocurre en la API."""
    host = _base(host)
    duration = max(4, min(15, int(duration)))
    content: list[dict] = [{"type": "text", "text": prompt}]
    for img in person_image_paths[:9]:
        content.append({"type": "image_url",
                        "image_url": {"url": _data_url(img)},
                        "role": "reference_image"})
    if video_path:
        clip = _trim_clip(video_path, duration)
        vurl = upload_media(clip, api_key, host)
        content.append({"type": "video_url", "video_url": {"url": vurl},
                        "role": "reference_video"})
    payload = {"model": MODEL, "content": content, "duration": duration,
               "resolution": resolution}
    if ratio != "adaptive":
        payload["ratio"] = ratio
    return _submit_and_wait(api_key, payload, poll_cb, host)


def _trim_clip(video_path: str, duration: int) -> str:
    """Recorta el video de referencia a `duration` s (H3 acepta 4-15 s)."""
    import subprocess
    from pathlib import Path

    out = str(Path(video_path).with_suffix("")) + f"_ref{duration}s.mp4"
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        ffmpeg = "ffmpeg"
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-t", str(duration), "-i", video_path,
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-an", out]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def download_to(url: str, dest: str) -> str:
    import httpx

    with httpx.stream("GET", url, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    return dest
