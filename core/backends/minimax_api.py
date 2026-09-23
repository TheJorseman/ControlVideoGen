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
    2067: "Limite de uso del Token Plan alcanzado (Hailuo): espera a que se resetee "
          "el cupo, sube de plan o compra Creditos. La peticion era valida.",
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
    """Sube un archivo y devuelve la referencia mm_file://{file_id}.

    purpose=video_generation_input es lo que espera la API para imagenes/video
    de referencia de generacion (valida specs al subir; valido 7 dias).
    """
    import httpx

    host = _base(host)
    from pathlib import Path

    with open(path, "rb") as fh:
        resp = httpx.post(
            f"{host}/v1/files/upload",
            headers=_headers(api_key),
            files={"file": (Path(path).name, fh)},
            data={"purpose": "video_generation_input"},
            timeout=300,
        )
    body = resp.json() if "json" in resp.headers.get("content-type", "") else {}
    _check(body, "files/upload")
    file_id = (body.get("file") or {}).get("file_id") or body.get("file_id")
    if not file_id:
        raise RuntimeError(f"No se recibio file_id: {str(body)[:250]}")
    return f"mm_file://{file_id}"


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
        clip, _meta = prepare_reference_media(video_path, max_duration=duration + 0.1)
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
    subprocess.run(cmd, check=True, capture_output=True, encoding="utf-8", errors="replace")
    return out


# ------------------------------------------------------------ preflight media
H264_CODECS = {"h264", "libx264", "hevc", "h265", "libx265"}


def _ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return "ffmpeg"


def probe_media(path: str) -> dict:
    """Duracion/codecs parseando 'ffmpeg -i path' (ffprobe no viene con imageio)."""
    import re
    import subprocess
    from pathlib import Path

    exe = _ffmpeg_exe()
    r = subprocess.run([exe, "-hide_banner", "-i", path],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    err = r.stderr or ""
    dur = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.?\d*)", err)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    vcodec = (re.search(r"Video:\s*(\w+)", err) or [None, None])[1]
    acodec = (re.search(r"Audio:\s*(\w+)", err) or [None, None])[1]
    try:
        size_mb = round(Path(path).stat().st_size / 1e6, 1)
    except OSError:
        size_mb = 0.0
    return {"duration": dur, "vcodec": vcodec, "acodec": acodec, "size_mb": size_mb}


def prepare_reference_media(path: str, max_duration: float = 15.0) -> tuple[str, dict]:
    """Deja el video conforme al estandar H3: <=max_duration s, H.264/HEVC, audio AAC.

    Replica el paso 'derivado conforme' del flujo del CLI: probe -> si VP9 u otro
    codec no admitido (o demasiado largo) transcode a libx264 CRF20 conservando
    el audio AAC. Devuelve (ruta_preparada, reporte_preflight).
    """
    import subprocess
    from pathlib import Path

    meta = probe_media(path)
    ok_codec = (meta["vcodec"] or "") in H264_CODECS
    ok_dur = 2.0 <= meta["duration"] <= max_duration
    if ok_codec and ok_dur:
        return path, {**meta, "prepared": False}

    exe = _ffmpeg_exe()
    src = Path(path)
    out = src.with_name(src.stem + "_prepared.mp4")
    dur = min(max_duration - 0.05, meta["duration"])
    cmd = [exe, "-y", "-loglevel", "error", "-t", f"{dur:.2f}", "-i", path,
           "-c:v", "libx264", "-preset", "slow", "-crf", "20",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, encoding="utf-8", errors="replace")
    return str(out), {**meta, "prepared": True, "new_duration": round(dur, 2)}


# ------------------------------------------------------------ Hailuo I2V (v1)
def hailuo_i2v(api_key: str, prompt: str, first_frame_path: str,
               duration: int = 6, resolution: str = "768P",
               model: str = "MiniMax-Hailuo-2.3", poll_cb=None,
               host: str = "") -> str:
    """Imagen-a-video first-frame con Hailuo (serie v1, elegible para Token Plan).

    El CLI replicate: sube la imagen, POST /v1/video_generation con
    first_frame_image, poll /v1/query/video_generation?task_id=, descarga via
    /v1/files/retrieve. Duracion maxima 10s a 768P. Sin audio nativo.
    """
    import time

    import httpx

    host = _base(host)
    if not api_key:
        raise RuntimeError("Falta la API key de MiniMax (Ajustes).")
    duration = max(6, min(10, int(duration)))
    payload = {
        "model": model,
        "prompt": prompt,
        "first_frame_image": _data_url(first_frame_path),
        "duration": duration,
        "resolution": resolution,
        "prompt_optimizer": False,
    }
    r = httpx.post(f"{host}/v1/video_generation", headers=_headers(api_key),
                   json=payload, timeout=120)
    body = r.json() if "json" in r.headers.get("content-type", "") else {}
    _check(body, "video_generation (hailuo)")
    task_id = body.get("task_id")
    if not task_id:
        raise RuntimeError(f"MiniMax no devolvio task_id: {str(body)[:250]}")
    file_id = None
    while True:
        time.sleep(10)
        if poll_cb:
            poll_cb("waiting")
        rr = httpx.get(f"{host}/v1/query/video_generation", headers=_headers(api_key),
                       params={"task_id": task_id}, timeout=60)
        rb = rr.json()
        _check(rb, "query (hailuo)")
        status = rb.get("status")
        if status == "Success" or rb.get("file_id"):
            file_id = rb.get("file_id")
            if file_id:
                break
        if status in ("Fail", "Timeout"):
            raise RuntimeError(f"Tarea hailuo {status}: {rb.get('base_resp')}")
    fr = httpx.get(f"{host}/v1/files/retrieve", headers=_headers(api_key),
                   params={"file_id": file_id}, timeout=60)
    fb = fr.json()
    _check(fb, "files/retrieve")
    vurl = (fb.get("file") or {}).get("download_url")
    if not vurl:
        raise RuntimeError(f"retrieve sin download_url: {str(fb)[:250]}")
    return vurl


# ------------------------------------------------------------ M3 auto-prompt
def describe_motion(api_key: str, frame_paths: list[str], host: str = "",
                    model: str = "MiniMax-M3") -> str:
    """M3 (Token Plan) entiende imagenes: convierte frames del video de
    referencia en un prompt de coreografia beat-by-beat para I2VA."""
    import base64
    from pathlib import Path

    import httpx

    host = _base(host)
    if not api_key:
        raise RuntimeError("Falta la API key de MiniMax (Ajustes).")
    blocks = [{
        "type": "text",
        "text": (
            "These are sequential frames from a dance video. Write an English "
            "prompt for an AI video generator describing the person's motion "
            "beat by beat, temporally ordered, 60-90 words, starting from "
            "'The person from the image ...'. NEVER use gendered pronouns "
            "(he/she/her/him) or describe appearance — only motion and pose. "
            "Mention camera as static. Output only the prompt."
        ),
    }]
    for fp in frame_paths:
        b = base64.b64encode(Path(fp).read_bytes()).decode()
        blocks.append({"type": "image", "source": {"type": "base64",
                       "media_type": "image/png", "data": b}})
    r = httpx.post(f"{host}/anthropic/v1/messages",
                   headers={"authorization": f"Bearer {api_key}",
                            "anthropic-version": "2023-06-01"},
                   json={"model": model, "max_tokens": 400,
                         "messages": [{"role": "user", "content": blocks}]},
                   timeout=180)
    body = r.json()
    if r.status_code >= 400:
        msg = (body.get("error") or {}).get("message", str(body)[:250])
        raise RuntimeError(f"M3 describe_motion HTTP {r.status_code}: {msg}")
    return (body.get("content") or [{}])[0].get("text", "").strip()


def build_i2va_prompt(motion_text: str) -> str:
    """Formato I2VA del CLI: ancla Picture 1 como primer frame + coreografia."""
    return (
        "For the target video, at 0.00 seconds, <Picture 1> is fully referenced "
        "as the first frame. "
        f"[Shot 1] {motion_text} "
        "overall_soundscape: N/A. non_diegetic_music: N/A."
    )


def download_to(url: str, dest: str) -> str:
    import httpx

    with httpx.stream("GET", url, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    return dest
