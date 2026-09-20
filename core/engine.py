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

# progreso compartido para la UI
PROGRESS: dict = {"stage": "", "step": 0, "total_steps": 0, "done": True}


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
    _CACHE["pipe"] = loader(local_dir, resident=resident)
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
    PROGRESS["step"] = step + 1
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

        PROGRESS.update({"stage": "MiniMax H3 (API): enviando tarea...", "step": 0,
                         "total_steps": 0, "done": False})
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUTPUT_DIR / f"minimax_{kind}_{stamp}.mp4"

        def poll_cb(_):
            PROGRESS["stage"] = "MiniMax H3 (API): generando en la nube (10s por poll)..."

        url = minimax_api.generate(
            api_key=settings["api_keys"].get("minimax", ""),
            prompt=prompt,
            kind=kind,
            image_path=image_path,
            last_image_path=last_image_path,
            duration=max(5, min(15, round(num_frames / 24))),
            resolution="768P",
            ratio=_ratio_from_size(width, height),
            poll_cb=poll_cb,
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

    PROGRESS.update(
        {"stage": "Cargando modelo (puede tardar la primera vez)...", "step": 0,
         "total_steps": steps, "done": False}
    )
    t0 = time.time()
    if backend == "ltx_25":
        pipe = _ensure_pipe(settings, ltx_be.MODEL_KEY, ltx_be.load_ltx)
        pipe._interrupt = False
        PROGRESS["stage"] = "Generando..."
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
        PROGRESS["stage"] = "Generando..."
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
        PROGRESS["stage"] = "Generando..."
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
}


def v2v_modes() -> list[tuple[str, str]]:
    return [(label, key) for key, (m, label) in _V2V_MODES.items()]


def _v2v_loader(model_key: str):
    from .backends import vace as vace_be, wan_animate as anim_be

    return anim_be.load_animate if model_key == anim_be.MODEL_KEY else vace_be.load_vace


def _snapped_frames(n: int) -> int:
    n = max(17, min(int(n), 257))
    return (n - 1) // 4 * 4 + 1


def prepare_conditions(settings: dict, video_path: str, mode: str,
                       resolution: str, max_frames: int):
    """Extrae las condiciones del video y guarda un preview. Devuelve dict con todo."""
    from . import preprocess

    height, width = _parse_size(resolution)
    n = _snapped_frames(min(max_frames, resolve_profile(settings)["max_frames"]))
    frames, fps = preprocess.load_video(video_path, max_frames=n, size=(width, height))
    if len(frames) < 17:
        raise RuntimeError("El video es demasiado corto (minimo 17 frames).")
    frames = frames[: _snapped_frames(len(frames))]

    conds = {"frames": frames, "fps": fps, "height": height, "width": width}
    pil = preprocess.frames_to_pil(frames)
    conds["pil"] = pil

    kind = _V2V_MODES[mode][0]
    if mode in ("animate_replace", "animate"):
        conds["pose"] = preprocess.extract_pose(frames, settings["model_dir"])
        if mode == "animate_replace":
            conds["mask"] = preprocess.extract_person_masks(frames)
        conds["face"] = preprocess.extract_face_crops(
            frames, settings["model_dir"], mask_frames=conds.get("mask")
        )
        preview = preprocess.save_video(
            preprocess.pil_to_frames(conds["pose"]), str(OUTPUT_DIR / "preview_pose.mp4"), fps
        )
    else:
        cond = (preprocess.extract_pose(frames, settings["model_dir"])
                if mode == "vace_pose" else preprocess.extract_depth(frames))
        conds["control"] = cond
        preview = preprocess.save_video(
            preprocess.pil_to_frames(cond), str(OUTPUT_DIR / "preview_cond.mp4"), fps
        )
    conds["preview"] = preview
    return conds


def generate_v2v(settings: dict, mode: str, video_path: str, prompt: str,
                 negative_prompt: str, resolution: str, max_frames: int,
                 steps: int, guidance: float, seed: int,
                 character_image_path: str | None = None) -> tuple[str, str]:
    """Flujo completo Video-a-Video. Devuelve (mp4_final, preview_condiciones)."""
    from . import preprocess
    from .backends import vace as vace_be, wan_animate as anim_be

    if mode not in _V2V_MODES:
        raise RuntimeError(f"Modo desconocido: {mode}")
    model_key = _V2V_MODES[mode][0]

    PROGRESS.update({"stage": "Extrayendo condiciones (pose/depth/mask)...",
                     "step": 0, "total_steps": 0, "done": False})
    conds = prepare_conditions(settings, video_path, mode, resolution, max_frames)
    n = len(conds["frames"])
    segments = max(1, -(-n // 77)) if model_key == "wan_animate" else 1
    PROGRESS["total_steps"] = int(steps) * segments

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
        frames_out = anim_be.run_animate(
            pipe,
            image=character,
            pose_frames=conds["pose"],
            face_frames=conds["face"],
            background_frames=conds["pil"] if mode == "animate_replace" else None,
            mask_frames=conds.get("mask") if mode == "animate_replace" else None,
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode="replace" if mode == "animate_replace" else "animate",
            height=conds["height"],
            width=conds["width"],
            steps=int(steps),
            guidance=float(guidance) if mode == "animate" else 1.0,
            seed=int(seed),
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
    PROGRESS.update({"stage": "Listo", "done": True})
    return str(out), conds["preview"]
