"""Motor: deteccion de GPU, perfiles de VRAM, cache de pipelines y generacion."""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from . import model_manager
from .config import OUTPUT_DIR, model_path

PROFILES = {
    "6gb": {"vram_gb": 6, "resident": False, "max_resolution": "480p", "max_frames": 81},
    "8gb": {"vram_gb": 8, "resident": False, "max_resolution": "480p", "max_frames": 81},
    "12gb": {"vram_gb": 12, "resident": False, "max_resolution": "720p", "max_frames": 121},
    "16gb": {"vram_gb": 16, "resident": False, "max_resolution": "720p", "max_frames": 121},
    "24gb": {"vram_gb": 24, "resident": True, "max_resolution": "720p", "max_frames": 121},
}

RESOLUTIONS = {
    "480p": ["832x480 (16:9)", "480x832 (9:16)", "624x624 (1:1)"],
    "720p": ["1280x720 (16:9)", "720x1280 (9:16)", "960x960 (1:1)"],
}

# {pipe, key} cache de un unico pipeline cargado
_CACHE: dict = {}

# progreso compartido para la UI y los logs
PROGRESS: dict = {"stage": "", "step": 0, "total_steps": 0, "segment": 0,
                  "total_segments": 1, "steps_per_seg": 0, "t0": 0.0,
                  "done": True, "pre_step": 0, "pre_total": 0}


def progress_reset(stage: str, steps: int = 0, segments: int = 1, sps: int = 0):
    PROGRESS.update({"stage": stage, "step": 0, "total_steps": steps,
                     "segment": 1, "total_segments": segments, "steps_per_seg": sps,
                     "t0": time.time(), "done": False, "pre_step": 0, "pre_total": 0})


def progress_state() -> dict:
    p = dict(PROGRESS)
    p.setdefault("avg_step", 0.0)
    if not p["done"] and p["total_steps"]:
        done_units = (p["segment"] - 1) * p["steps_per_seg"] + p["step"]
        done_units = min(done_units, p["total_steps"])
        elapsed = max(time.time() - p["t0"], 0.001)
        avg = elapsed / max(done_units, 1)
        p.update({"fraction": done_units / p["total_steps"],
                  "done_units": done_units, "avg_step": avg,
                  "eta": avg * (p["total_steps"] - done_units)})
    elif p["done"]:
        p["fraction"] = 1.0 if p["total_steps"] else 0.0
        p["eta"] = 0
        p["done_units"] = p["total_steps"]
    else:
        p.update({"fraction": 0.0, "eta": 0, "done_units": 0})
    return p


def detect_gpu() -> dict:
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "name": props.name,
                "vram_gb": round(props.total_memory / 1e9),
                "torch_cuda": torch.version.cuda,
            }
    except Exception as exc:  # noqa: BLE001
        return {"name": f"sin GPU ({exc})", "vram_gb": 0, "torch_cuda": "-"}
    return {"name": "sin GPU", "vram_gb": 0, "torch_cuda": "-"}


def resolve_profile(settings: dict) -> dict:
    name = settings.get("vram_profile", "auto")
    if name in PROFILES:
        return PROFILES[name]
    gpu_gb = detect_gpu()["vram_gb"]
    best, prev = None, None
    for key, prof in PROFILES.items():
        if prof["vram_gb"] <= max(gpu_gb, 6):
            best, prev = key, prof
    if best:
        return PROFILES[best]
    return prev or PROFILES["6gb"]


def resolution_options(settings: dict) -> list[str]:
    prof = resolve_profile(settings)
    tiers = ["480p"] if prof["max_resolution"] == "480p" else ["480p", "720p"]
    out = []
    for tier in tiers:
        out.extend(RESOLUTIONS[tier])
    return out


def max_frames(settings: dict) -> int:
    return resolve_profile(settings)["max_frames"]


AUTO_RESOLUTION = "Auto (respeta orientacion del video)"


def _auto_size(src_w: int, src_h: int, settings: dict, heavy: bool = False) -> str:
    """Elige resolucion del catalogo segun la orientacion del video de origen.
    heavy=True (modelos 14B): se limita al tier 480p para caber en 16 GB."""
    prof = resolve_profile(settings)
    portrait = src_h > src_w
    if prof["max_resolution"] == "720p" and not heavy:
        return "720x1280 (auto 9:16)" if portrait else "1280x720 (auto 16:9)"
    return "480x832 (auto 9:16)" if portrait else "832x480 (auto 16:9)"


def _parse_size(text: str) -> tuple[int, int]:
    wh = text.split()[0]
    w, h = wh.split("x")
    return int(h), int(w)


def _ensure_pipe(settings: dict, model_key: str, loader, cache_key: str | None = None):
    """Carga (y cachea) un pipeline bajo la clave de cache dada."""
    cache_key = cache_key or model_key
    if _CACHE.get("cache_key") == cache_key and _CACHE.get("pipe") is not None:
        return _CACHE["pipe"]
    spec = model_manager.REGISTRY[model_key]
    local_dir = str(model_path(settings["model_dir"], model_key))
    if not Path(local_dir).exists():
        raise RuntimeError(
            f"No descargado: {spec.label}. Ve a Modelos y Ajustes para descargarlo."
        )
    prof = resolve_profile(settings)
    resident = prof["resident"] and spec.approx_gb < 15
    kwargs = {"resident": resident}
    import inspect

    if "quantize" in inspect.signature(loader).parameters:
        # Modelos grandes (>15 GB): INT8 + streaming por bloques en GPUs de consumo
        kwargs["quantize"] = spec.approx_gb > 15 and prof["vram_gb"] < 32
    _CACHE["pipe"] = loader(local_dir, **kwargs)
    _CACHE["cache_key"] = cache_key
    _CACHE["model_key"] = model_key
    return _CACHE["pipe"]


def unload() -> str:
    _CACHE.clear()
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return "Modelo descargado de memoria."


def interrupt() -> str:
    pipe = _CACHE.get("pipe")
    if pipe is not None:
        pipe._interrupt = True
        return "Deteniendo tras el paso actual..."
    return "No hay generacion en curso."


def _step_callback(_pipe, step: int, _t, cb_kwargs):
    p = PROGRESS
    if p["total_segments"] > 1 and step == 0 and p["step"] >= p["steps_per_seg"] - 1:
        p["segment"] += 1
    p["step"] = min(step + 1, p["steps_per_seg"] or step + 1)
    s = progress_state()
    print(f"[progreso] {p['stage']} · seg {p['segment']}/{p['total_segments']} "
          f"paso {p['step']}/{p['steps_per_seg'] or '?'} · "
          f"{s['fraction'] * 100:.0f}% · {s['avg_step']:.0f}s/paso · "
          f"ETA {int(s['eta'] // 60)}m{int(s['eta'] % 60):02d}s", flush=True)
    return cb_kwargs


BACKEND_CHOICES = [
    ("Wan 2.2 TI2V-5B (local)", "wan_5b"),
    ("LTX-2.5 distilled (local, gated HF)", "ltx_25"),
    ("MiniMax H3 (API)", "minimax_api"),
]


def _ratio_from_size(width: int, height: int) -> str:
    r = width / height
    if r > 1.5:
        return "21:9" if r > 1.9 else "16:9"
    if r < 0.67:
        return "9:16"
    if r < 0.95:
        return "3:4"
    if r <= 1.1:
        return "1:1"
    return "4:3"


def generate(
    settings: dict,
    kind: str,
    prompt: str,
    negative_prompt: str,
    resolution: str,
    num_frames: int,
    steps: int,
    guidance: float,
    seed: int,
    image_path: str | None = None,
    last_image_path: str | None = None,
    backend: str = "wan_5b",
) -> str:
    """Genera un video. kind: 't2v'|'i2v'. Devuelve ruta mp4."""
    from .backends import ltx as ltx_be, wan as wan_be

    prof = resolve_profile(settings)
    num_frames = min(int(num_frames), prof["max_frames"])
    height, width = _parse_size(resolution)

    # ------------------------------------------------ MiniMax H3 via API
    if backend == "minimax_api":
        from .backends import minimax_api

        progress_reset("MiniMax H3 (API): enviando tarea...")

        def _tick_poll():
            s = time.time() - PROGRESS["t0"]
            PROGRESS["stage"] = f"MiniMax H3 (API): generando en la nube ({int(s // 60)}m{int(s % 60):02d}s)"

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / f"minimax_{kind}_{stamp}.mp4"

        url = minimax_api.generate(
            api_key=settings["api_keys"].get("minimax", ""),
            prompt=prompt,
            kind=kind,
            image_path=image_path,
            last_image_path=last_image_path,
            duration=max(5, min(15, round(num_frames / 24))),
            resolution="768P",
            ratio=_ratio_from_size(width, height),
            poll_cb=_tick_poll,
            host=settings.get("minimax_host", ""),
        )
        minimax_api.download_to(url, str(out))
        PROGRESS.update({"stage": "Listo", "done": True})
        return str(out)

    image = last_image = None
    if kind == "i2v":
        from PIL import Image

        if not image_path:
            raise RuntimeError("Sube una imagen de inicio.")
        image = Image.open(image_path).convert("RGB")
        if last_image_path:
            last_image = Image.open(last_image_path).convert("RGB")

    t0 = time.time()
    progress_reset("Cargando modelo (puede tardar la primera vez)...",
                   steps=int(steps), segments=1, sps=int(steps))
    if backend == "ltx_25":
        pipe = _ensure_pipe(settings, ltx_be.MODEL_KEY, ltx_be.load_ltx)
        pipe._interrupt = False
        PROGRESS.update({"stage": "Generando...", "t0": time.time(), "step": 0,
                         "segment": 1})
        frames = ltx_be.run_ltx(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            steps=int(steps),
            guidance=float(guidance),
            seed=int(seed),
            step_callback=_step_callback,
        )
    elif kind == "i2v":
        pipe = _ensure_pipe(settings, wan_be.MODEL_KEY, wan_be.load_i2v,
                            cache_key=f"{wan_be.MODEL_KEY}:i2v")
        pipe._interrupt = False
        PROGRESS.update({"stage": "Generando...", "t0": time.time(), "step": 0,
                         "segment": 1})
        frames = wan_be.run_i2v(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            height=height,
            width=width,
            num_frames=num_frames,
            steps=int(steps),
            guidance=float(guidance),
            seed=int(seed),
            last_image=last_image,
            step_callback=_step_callback,
        )
    else:
        pipe = _ensure_pipe(settings, wan_be.MODEL_KEY, wan_be.load_t2v,
                            cache_key=f"{wan_be.MODEL_KEY}:t2v")
        pipe._interrupt = False
        PROGRESS.update({"stage": "Generando...", "t0": time.time(), "step": 0,
                         "segment": 1})
        frames = wan_be.run_t2v(
            pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            steps=int(steps),
            guidance=float(guidance),
            seed=int(seed),
            step_callback=_step_callback,
        )

    from diffusers.utils import export_to_video

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"{kind}_{stamp}.mp4"
    export_to_video(frames, str(out), fps=24)

    PROGRESS.update({"stage": f"Listo en {int(time.time() - t0)} s", "done": True})
    return str(out)


def disk_report(settings: dict) -> str:
    free = model_manager.disk_free_gb(settings["model_dir"])
    used = model_manager.dir_size_gb(Path(settings["model_dir"]))
    return f"Directorio: {settings['model_dir']}  |  Usado: {used:.1f} GB  |  Libre en disco: {free:.1f} GB"


# ============================================================== V2V (Fase 2)
_V2V_MODES = {
    "animate_replace": ("wan_animate", "Wan Animate: reemplazar persona conservando fondo"),
    "animate": ("wan_animate", "Wan Animate: animar personaje con la pose del video"),
    "vace_pose": ("wan_vace", "VACE: video de esqueleto pose + prompt"),
    "vace_depth": ("wan_vace", "VACE: video de profundidad + prompt"),
    "minimax_r2va": ("minimax_api", "MiniMax H3 (nube, sin GPU): video de motion + foto de la persona"),
    "minimax_i2v": ("minimax_i2v", "MiniMax Hailuo-2.3 (nube, TOKEN PLAN): imagen como primer frame + prompt de motion + audio del original"),
}


def v2v_modes() -> list[tuple[str, str]]:
    return [(label, key) for key, (m, label) in _V2V_MODES.items()]


def _v2v_loader(model_key: str):
    from .backends import vace as vace_be, wan_animate as anim_be

    return anim_be.load_animate if model_key == anim_be.MODEL_KEY else vace_be.load_vace


def _v2v_hailuo(settings: dict, video_path: str, prompt: str,
                character_image_path: str | None, duration_s: int = 10) -> tuple[str, str]:
    """Hailuo-2.3 (Token Plan): la imagen de referencia es el PRIMER FRAME y el
    prompt describe el motion beat-by-beat. Max 10 s @768P. H2.3 es mudo: se le
    muxea el audio del video original recortado a la duracion generada."""
    from .backends import minimax_api

    if not character_image_path:
        raise RuntimeError("Sube la imagen (sera el primer frame del video).")
    if not prompt.strip():
        raise RuntimeError("Escribe el prompt de motion (o usa el auto-prompt M3).")
    api_key = settings["api_keys"].get("minimax", "")
    host = settings.get("minimax_host", "")
    duration = 6 if int(duration_s) <= 6 else 10
    final_prompt = prompt if "is fully referenced" in prompt else \
        minimax_api.build_i2va_prompt(prompt)

    progress_reset(f"Hailuo-2.3 (nube): enviando imagen ({duration}s)...", steps=0)

    def _poll(_):
        s = time.time() - PROGRESS["t0"]
        PROGRESS["stage"] = (f"Hailuo-2.3 (nube): generando "
                             f"({int(s // 60)}m{int(s % 60):02d}s)")

    url = minimax_api.hailuo_i2v(
        api_key=api_key, prompt=final_prompt, first_frame_path=character_image_path,
        duration=duration, resolution="768P", poll_cb=_poll, host=host,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"v2v_hailuo_{stamp}.mp4"
    minimax_api.download_to(url, str(out))

    PROGRESS["stage"] = "Alineando audio del original al motion generado..."
    final, note = sync_generated_audio(str(out), video_path, settings)
    PROGRESS.update({"stage": f"Listo (nube, {note})", "done": True})
    return final, ""


def sync_generated_audio(generated_path: str, source_path: str,
                         settings: dict | None = None, threshold: float = 0.4):
    """Herramienta: detecta el offset de motion entre el clip generado y el video
    fuente y re-muxea el audio compensado. Devuelve (ruta_final, nota)."""
    import os
    from datetime import datetime

    from . import analysis
    from . import preprocess

    try:
        r = analysis.find_audio_offset(generated_path, source_path)
    except Exception:  # noqa: BLE001
        r = {"offset_s": 0.0, "confidence": 0.0}
    if r["confidence"] >= threshold and abs(r["offset_s"]) >= 0.1:
        new = os.path.join(
            str(OUTPUT_DIR),
            f"synced_{datetime.now().strftime('%H%M%S')}{os.path.splitext(os.path.basename(generated_path))[0][-24:]}.mp4",
        )
        analysis.mux_audio_aligned(generated_path, source_path, r["offset_s"], new)
        return new, (f"audio sincronizado {r['offset_s']:+.2f}s "
                     f"(conf {r['confidence']:.2f})")
    # fallback: mux simple sin offset
    new = generated_path.replace(".mp4", "_mux.mp4")
    try:
        import shutil

        shutil.copy(generated_path, new)
        preprocess._mux_audio(new, source_path)
        return new, f"audio mux sin offset (sync flojo: conf {r['confidence']:.2f})"
    except Exception:  # noqa: BLE001
        return generated_path, "sin audio"


def describe_motion(settings: dict, video_path: str, n_frames: int = 8,
                    target_duration: int = 10) -> str:
    """Auto-prompt: muestrea N frames UNIFORMEMENTE a lo largo de todo el video y
    M3 (Token Plan) devuelve una coreografia con timestamps beat-by-beat."""
    from .backends import minimax_api

    import cv2
    import numpy as np
    import PIL.Image

    if not video_path:
        raise RuntimeError("Sube primero el video de referencia.")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = np.linspace(0, total - 1, n_frames).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok:
            frames.append((int(i), cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError("No se pudieron leer frames del video.")

    tmp_dir = OUTPUT_DIR / "_m3_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for fidx, _fr in frames:
        PIL.Image.fromarray(_fr).resize((512, 896)).save(tmp_dir / f"f{fidx}.png")
    paths = [str(tmp_dir / f"f{i[0]}.png") for i in frames]

    PROGRESS.update({"stage": "M3 analizando los frames...", "done": False})
    try:
        text = minimax_api.describe_motion(
            settings["api_keys"].get("minimax", ""), paths,
            host=settings.get("minimax_host", ""), duration_s=target_duration,
        )
    finally:
        PROGRESS.update({"stage": "Auto-prompt listo", "done": True})
    if not text:
        raise RuntimeError("M3 no devolvio descripcion")
    return text


def _v2v_minimax(settings: dict, video_path: str, prompt: str, max_frames: int,
                 character_image_path: str | None,
                 start_reference: bool = False) -> tuple[str, str]:
    """Swap en la nube con H3 ref2va: motion del video + identidad de la foto."""
    from .backends import minimax_api

    if not character_image_path:
        raise RuntimeError("Sube la imagen de la persona nueva (o genererala arriba).")
    if not prompt.strip():
        raise RuntimeError("Escribe un prompt describiendo la escena/accion del video final.")
    api_key = settings["api_keys"].get("minimax", "")
    host = settings.get("minimax_host", "")
    duration = max(4, min(15, round(int(max_frames) / 24)))

    progress_reset(f"MiniMax (nube): subiendo referencia ({duration}s)...", steps=0)

    def _poll(_):
        s = time.time() - PROGRESS["t0"]
        PROGRESS["stage"] = (f"MiniMax (nube): generando en el servidor "
                             f"({int(s // 60)}m{int(s % 60):02d}s)")

    url = minimax_api.ref2va_generate(
        api_key=api_key, prompt=prompt,
        person_image_paths=[character_image_path], video_path=video_path,
        duration=duration, poll_cb=_poll, host=host,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"v2v_minimax_{stamp}.mp4"
    minimax_api.download_to(url, str(out))
    if start_reference:
        _start_frame_fade(str(out), character_image_path, fps=24)
    PROGRESS.update({"stage": "Listo (nube)", "done": True})
    return str(out), ""


def _snapped_frames(n: int) -> int:
    n = max(17, min(int(n), 257))
    return (n - 1) // 4 * 4 + 1


def prepare_conditions(settings: dict, video_path: str, mode: str,
                       resolution: str, max_frames: int):
    """Extrae las condiciones del video y guarda un preview. Devuelve dict con todo."""
    from . import preprocess

    kind = _V2V_MODES[mode][0]
    heavy = kind in ("wan_animate", "wan_vace")

    if resolution == AUTO_RESOLUTION:
        sw, sh, _fps, _n = preprocess.probe_video(video_path)
        resolution = _auto_size(sw, sh, settings, heavy=heavy)
    height, width = _parse_size(resolution)
    # los modelos 14B generan por segmentos -> hasta 257 frames; el resto, el techo del perfil
    cap = 257 if heavy else resolve_profile(settings)["max_frames"]
    n = _snapped_frames(min(max_frames, cap))

    progress_reset("Extrayendo frames del video...", steps=0)

    def _mk(stage):
        def cb(i, total):
            PROGRESS["stage"] = stage
            PROGRESS["pre_step"] = i
            PROGRESS["pre_total"] = total
            if i == total or i % 10 == 0:
                print(f"[progreso] {stage} {i}/{total}", flush=True)
        return cb

    frames, fps = preprocess.load_video(video_path, max_frames=n, size=(width, height),
                                        target_fps=24.0)
    if len(frames) < 17:
        raise RuntimeError("El video es demasiado corto (minimo 17 frames).")
    frames = frames[: _snapped_frames(len(frames))]

    conds = {"frames": frames, "fps": fps, "height": height, "width": width,
             "resolution": resolution}
    pil = preprocess.frames_to_pil(frames)
    conds["pil"] = pil

    stamp = datetime.now().strftime("%H%M%S")
    if mode in ("animate_replace", "animate"):
        conds["pose"] = preprocess.extract_pose(
            frames, settings["model_dir"], progress_cb=_mk("Pose (MediaPipe)")
        )
        if mode == "animate_replace":
            conds["mask"] = preprocess.extract_person_masks(
                frames, progress_cb=_mk("Mascara de persona (rembg)")
            )
        conds["face"] = preprocess.extract_face_crops(
            frames, settings["model_dir"], mask_frames=conds.get("mask"),
            progress_cb=_mk("Recortes de cara"),
        )
        preview = preprocess.save_video(
            preprocess.pil_to_frames(conds["pose"]),
            str(OUTPUT_DIR / f"preview_pose_{stamp}.mp4"), fps
        )
    else:
        if mode == "vace_pose":
            cond = preprocess.extract_pose(frames, settings["model_dir"],
                                           progress_cb=_mk("Pose (MediaPipe)"))
        else:
            cond = preprocess.extract_depth(frames, progress_cb=_mk("Depth (Depth Anything)"))
        conds["control"] = cond
        preview = preprocess.save_video(
            preprocess.pil_to_frames(cond),
            str(OUTPUT_DIR / f"preview_cond_{stamp}.mp4"), fps
        )
    conds["preview"] = preview
    return conds


def _start_frame_fade(video_path: str, ref_image_path: str, fade_s: float = 0.5,
                      fps: float | None = None) -> str:
    """Reescribe el video para que arranque en la imagen de referencia con un
    crossfade suave hacia el primer frame generado (empalma identidad y pose)."""
    import os

    import cv2
    import numpy as np
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    fps = fps or (cap.get(cv2.CAP_PROP_FPS) or 24.0)
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        return video_path

    h, w = frames[0].shape[:2]
    im = Image.open(ref_image_path).convert("RGB")
    src_ar, dst_ar = im.width / im.height, w / h
    if src_ar > dst_ar:  # crop laterales
        nw = int(im.height * dst_ar)
        x = (im.width - nw) // 2
        im = im.crop((x, 0, x + nw, im.height))
    elif src_ar < dst_ar:  # crop vertical
        nh = int(im.width / dst_ar)
        y = (im.height - nh) // 2
        im = im.crop((0, y, im.width, y + nh))
    ref = np.asarray(im.resize((w, h), Image.LANCZOS))

    k = max(2, int(round(fade_s * fps)))
    k = min(k, max(len(frames) // 4, 2))
    out_frames = []
    for i, fr in enumerate(frames):
        if i < k:
            a = (i + 1) / (k + 1)
            fr = (fr.astype(np.float32) * a + ref.astype(np.float32) * (1 - a)
                  ).astype(np.uint8)
        out_frames.append(fr)

    tmp = video_path + ".src.mp4"
    os.replace(video_path, tmp)
    try:
        from . import preprocess
        preprocess.save_video(np.stack(out_frames), video_path, fps, audio_from=tmp)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return video_path


def generate_v2v(settings: dict, mode: str, video_path: str, prompt: str,
                 negative_prompt: str, resolution: str, max_frames: int,
                 steps: int, guidance: float, seed: int,
                 character_image_path: str | None = None,
                 start_reference: bool = False,
                 duration_s: int = 10) -> tuple[str, str]:
    """Flujo completo Video-a-Video. Devuelve (mp4_final, preview_condiciones)."""
    from . import preprocess
    from .backends import vace as vace_be, wan_animate as anim_be

    if mode not in _V2V_MODES:
        raise RuntimeError(f"Modo desconocido: {mode}")
    model_key = _V2V_MODES[mode][0]

    if model_key == "minimax_api":
        return _v2v_minimax(settings, video_path, prompt, max_frames,
                            character_image_path, start_reference)
    if model_key == "minimax_i2v":
        return _v2v_hailuo(settings, video_path, prompt, character_image_path,
                           duration_s)

    conds = prepare_conditions(settings, video_path, mode, resolution, max_frames)
    n = len(conds["frames"])
    segments = max(1, -(-n // 77)) if model_key == "wan_animate" else 1
    progress_reset(f"Generando V2V ({mode})...", steps=int(steps) * segments,
                   segments=segments, sps=int(steps))

    if _CACHE.get("model_key") != model_key or _CACHE.get("pipe") is None:
        _CACHE["pipe"] = None
    pipe = _ensure_pipe(settings, model_key, _v2v_loader(model_key))
    pipe._interrupt = False

    PROGRESS["stage"] = "Generando video..."
    if model_key == "wan_animate":
        from PIL import Image

        if not character_image_path:
            raise RuntimeError("Sube la imagen de la persona/personaje nuevo.")
        character = Image.open(character_image_path).convert("RGB")
        bg_frames = conds["pil"]
        face_frames = conds["face"]
        if start_reference:
            # El frame 0 se fuerza como ancla del pipeline y el face_encoder es el
            # que mas peso tiene en la identidad: usamos la imagen de referencia como
            # frame 0 Y su cara como condicion facial, para no arrastrar la cara del
            # video original (que dominaba y revertia la identidad).
            im = character
            src_ar, dst_ar = im.width / im.height, conds["width"] / conds["height"]
            if src_ar > dst_ar:
                nw = int(im.height * dst_ar)
                x = (im.width - nw) // 2
                im = im.crop((x, 0, x + nw, im.height))
            elif src_ar < dst_ar:
                nh = int(im.width / dst_ar)
                y = (im.height - nh) // 2
                im = im.crop((0, y, im.width, y + nh))
            bg_frames = [im.resize((conds["width"], conds["height"]), Image.LANCZOS)] + list(bg_frames[1:])
            import numpy as np

            char_face = preprocess.extract_face_crops(np.asarray(character)[np.newaxis])
            face_frames = [char_face[0]] * n
        frames_out = anim_be.run_animate(
            pipe,
            image=character,
            pose_frames=conds["pose"],
            face_frames=face_frames,
            background_frames=bg_frames if mode == "animate_replace" else None,
            mask_frames=conds.get("mask") if mode == "animate_replace" else None,
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode="replace" if mode == "animate_replace" else "animate",
            height=conds["height"],
            width=conds["width"],
            steps=int(steps),
            guidance=float(guidance) if mode == "animate" else 1.0,
            seed=int(seed),
            prev_segment_conditioning_frames=5 if start_reference else 1,
            step_callback=_step_callback,
        )
    else:
        reference = None
        if character_image_path:
            from PIL import Image

            reference = Image.open(character_image_path).convert("RGB")
        frames_out = vace_be.run_vace(
            pipe,
            control_frames=conds["control"],
            prompt=prompt,
            negative_prompt=negative_prompt,
            reference_image=reference,
            height=conds["height"],
            width=conds["width"],
            num_frames=n,
            steps=int(steps),
            guidance=float(guidance),
            seed=int(seed),
            step_callback=_step_callback,
        )

    from diffusers.utils import export_to_video

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUTPUT_DIR / f"v2v_{mode}_{stamp}.mp4"
    export_to_video(frames_out, str(out), fps=24)
    preprocess._mux_audio(str(out), video_path)
    if start_reference and character_image_path:
        _start_frame_fade(str(out), character_image_path, fps=24)
    PROGRESS.update({"stage": "Listo", "done": True})
    return str(out), conds["preview"]
