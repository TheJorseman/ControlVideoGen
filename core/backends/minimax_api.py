"""Backend MiniMax H3 via API oficial (POST /v2/video_generation + polling).

Requisitos: API key de MiniMax (plan pay-as-you-go) en Ajustes.
"""
from __future__ import annotations

BASE = "https://api.minimax.io"
MODEL = "MiniMax-H3"


def _headers(api_key: str) -> dict:
    return {"authorization": f"Bearer {api_key}"}


def upload_media(path: str, api_key: str) -> str:
    """Sube un archivo local y devuelve su URL de descarga (API de ficheros v1)."""
    import httpx

    with open(path, "rb") as fh:
        resp = httpx.post(
            f"{BASE}/v1/files/upload",
            headers=_headers(api_key),
            files={"file": (path.rsplit("/", 1)[-1], fh)},
            data={"purpose": "video-generation"},
            timeout=300,
        )
    resp.raise_for_status()
    body = resp.json()
    base_resp = body.get("base_resp") or {}
    if base_resp.get("status_code") not in (0, None):
        raise RuntimeError(f"MiniMax upload: {base_resp}")
    file_id = (body.get("file") or {}).get("file_id") or body.get("file_id")
    if not file_id:
        raise RuntimeError(f"No se recibio file_id: {body}")
    resp = httpx.get(f"{BASE}/v1/files/retrieve", headers=_headers(api_key),
                     params={"file_id": file_id}, timeout=60)
    resp.raise_for_status()
    url = (resp.json().get("file") or {}).get("download_url")
    if not url:
        raise RuntimeError(f"MiniMax retrieve sin download_url: {resp.json()}")
    return url


def _build_content(prompt: str, image_path: str | None, last_image_path: str | None,
                   api_key: str) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    if image_path:
        url = upload_media(image_path, api_key)
        content.append({"type": "image_url", "image_url": {"url": url},
                        "role": "first_frame"})
    if last_image_path:
        url = upload_media(last_image_path, api_key)
        content.append({"type": "image_url", "image_url": {"url": url},
                        "role": "last_frame"})
    return content


def generate(api_key: str, prompt: str, kind: str = "t2v",
             image_path: str | None = None, last_image_path: str | None = None,
             duration: int = 5, resolution: str = "768P", ratio: str = "16:9",
             poll_cb=None) -> str:
    """Crea la tarea, hace polling y devuelve la URL del video final."""
    import httpx

    if not api_key:
        raise RuntimeError("Falta la API key de MiniMax (Ajustes).")
    content = _build_content(prompt, image_path if kind == "i2v" else None,
                             last_image_path if kind == "i2v" else None, api_key)
    payload = {"model": MODEL, "content": content, "duration": int(duration),
               "resolution": resolution}
    if kind == "t2v":
        payload["ratio"] = ratio

    resp = httpx.post(f"{BASE}/v2/video_generation", headers=_headers(api_key),
                      json=payload, timeout=120)
    resp.raise_for_status()
    task_id = resp.json().get("task_id") or resp.json().get("task", {}).get("id")
    if not task_id:
        raise RuntimeError(f"MiniMax no devolvio task_id: {resp.json()}")

    import time

    while True:
        time.sleep(10)
        if poll_cb:
            poll_cb("waiting")
        r = httpx.get(f"{BASE}/v2/query/video_generation/{task_id}",
                      headers=_headers(api_key), timeout=60)
        r.raise_for_status()
        task = r.json()["task"]
        status = task["status"]
        if status == "succeeded":
            return task["content"]["url"]
        if status in ("failed", "cancelled"):
            raise RuntimeError(f"Tarea {status}: {task.get('error')}")


def download_to(url: str, dest: str) -> str:
    import httpx

    with httpx.stream("GET", url, timeout=300) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_bytes():
                fh.write(chunk)
    return dest
