"""Analisis motion-to-audio: alinea el audio del video original al baile GENERADO.

Hailuo-2.3 inventa la coreografia desde el texto, no copia los tiempos del video
fuente, asi que el audio original queda desfaseado ('semantico'). Este módulo:

1. motion_energy(): curva de energia de movimiento (diff de frames consecutivos)
   de un video, remuestreada a un fps comun.
2. find_audio_offset(): correlaciona las curvas del video generado y del video
   fuente y devuelve el offset (segundos) al que el audio debe desplazarse para
   que los golpes de baile coincidan.
3. mux_audio_aligned(): muxea el audio del source al generado aplicando ese
   offset (retardo o recorte) y la duracion del clip.

Todo con OpenCV + numpy + ffmpeg (imageio-ffmpeg), sin dependencias extra.
"""
from __future__ import annotations

import numpy as np

from .backends.minimax_api import _ffmpeg_exe


def _motion_curve(path: str, sample_fps: float = 10.0, max_seconds: float | None = None):
    """Devuelve (times, energy) normalizada 0-1 de movimiento por frame."""
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    if fps <= 0:
        fps = 24.0
    step = max(1, round(fps / sample_fps))
    energies = []
    prev = None
    idx = 0
    t = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            g = cv2.cvtColor(cv2.resize(frame, (128, 128)), cv2.COLOR_BGR2GRAY)
            g = g.astype(np.float32)
            if prev is not None:
                energies.append(float(np.abs(g - prev).mean()))
                t = len(energies) / sample_fps
            prev = g
        idx += 1
        if max_seconds and idx / fps > max_seconds:
            break
    cap.release()
    if not energies:
        return np.array([]), np.array([])
    e = np.array(energies, dtype=np.float32)
    if e.max() > 0:
        e = e / e.max()
    times = np.arange(len(e)) / sample_fps
    return times, e


def _interp(times, curve, new_times):
    if len(times) == 0:
        return np.zeros_like(new_times)
    return np.interp(new_times, times, curve, left=0.0, right=0.0)


def find_audio_offset(generated_path: str, source_path: str,
                      max_offset: float = 3.0, fps: float = 10.0,
                      min_confidence: float = 0.4) -> dict:
    """Offset (segundos) al que mover el audio para que el motion coincida.

    positivo => retrasar el audio; negativo => adelantarlo (recortar inicio).
    Devuelve dict con offset_s, confidence (pico de correlacion normalizada).
    Clips < 4 s o correlaciones < min_confidence no son fiables => offset 0."""
    gt, ge = _motion_curve(generated_path, sample_fps=fps)
    st, se = _motion_curve(source_path, sample_fps=fps)
    if len(ge) < 4 or len(se) < 4:
        return {"offset_s": 0.0, "confidence": 0.0}
    if gt[-1] < 4.0:  # clip demasiado corto: la correlacion es ruido
        return {"offset_s": 0.0, "confidence": 0.0}

    gen_len = gt[-1] + 1.0 / fps
    # grid comun sobre la duracion del generado
    grid = np.arange(0, gen_len, 1.0 / fps)
    g = _interp(gt, ge, grid)
    g = g - g.mean()

    best = {"offset_s": 0.0, "confidence": -2.0}
    offsets = np.arange(-max_offset, max_offset + 1e-6, 0.1)
    for off in offsets:
        s = _interp(st, se, grid + off)
        s = s - s.mean()
        denom = (np.linalg.norm(g) * np.linalg.norm(s)) + 1e-8
        corr = float((g * s).sum() / denom)
        if corr > best["confidence"]:
            best = {"offset_s": round(float(off), 2), "confidence": round(corr, 3)}
    if best["confidence"] < min_confidence:
        return {"offset_s": 0.0, "confidence": best["confidence"]}
    if abs(best["offset_s"]) > gen_len * 0.5:  # un desplazamiento mayor que el
        return {"offset_s": 0.0, "confidence": best["confidence"]}  # propio clip es absurdo
    return best


def mux_audio_aligned(generated_path: str, audio_source_path: str, offset_s: float,
                      output_path: str | None = None) -> str:
    """Copia el video generado y le pone el audio del source desplazado offset_s.

    offset_s > 0  -> el audio empieza offset_s despues (retraso/adelay).
    offset_s < 0  -> se recortan |offset_s| segundos del inicio del audio.
    La duracion final es la del video generado (-shortest)."""
    import subprocess
    from pathlib import Path

    out = output_path or (str(Path(generated_path).with_suffix("")) + "_synced.mp4")
    exe = _ffmpeg_exe()
    cmd = [exe, "-y", "-loglevel", "error", "-i", generated_path]
    if offset_s >= 0:
        cmd += ["-i", audio_source_path]
    else:
        cmd += ["-ss", f"{abs(offset_s):.3f}", "-i", audio_source_path]
    if offset_s > 0:
        ms = int(offset_s * 1000)
        cmd += ["-filter:a", f"adelay={ms}|{ms}"]
    cmd += ["-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac",
            "-b:a", "160k", "-shortest", out]
    subprocess.run(cmd, check=True, capture_output=True,
                   encoding="utf-8", errors="replace")
    return out
