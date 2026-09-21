"""Gestor de modelos: registro, descarga, inventario y borrado de pesos.

Cada modelo se guarda en <model_dir>/<clave> via snapshot_download(local_dir=...),
de modo que diffusers puede cargarlo directamente desde esa carpeta offline.
"""
from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import model_path


@dataclass
class ModelSpec:
    key: str
    label: str
    repo_id: str
    tasks: str
    approx_gb: float
    min_vram_gb: float
    note: str = ""
    phase: int = 1
    allow_patterns: tuple = field(default_factory=tuple)
    direct_url: str = ""
    filename: str = ""


REGISTRY: dict[str, ModelSpec] = {
    "wan_ti2v_5b": ModelSpec(
        key="wan_ti2v_5b",
        label="Wan 2.2 TI2V-5B (rapido, 720p@24fps)",
        repo_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        tasks="T2V, I2V, V2V-light",
        approx_gb=11.0,
        min_vram_gb=6,
        note="Recomendado para equipos con poca VRAM. Un solo modelo hace texto->video e imagen->video.",
        phase=1,
    ),
    "wan_t2v_14b": ModelSpec(
        key="wan_t2v_14b",
        label="Wan 2.2 T2V-A14B (maxima calidad)",
        repo_id="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        tasks="T2V",
        approx_gb=60.0,
        min_vram_gb=12,
        note="MoE 27B. Requiere offload intenso en 16GB; usar solo si buscas calidad maxima.",
        phase=2,
    ),
    "wan_i2v_14b": ModelSpec(
        key="wan_i2v_14b",
        label="Wan 2.2 I2V-A14B (imagen a video 14B)",
        repo_id="Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        tasks="I2V",
        approx_gb=60.0,
        min_vram_gb=12,
        phase=2,
    ),
    "wan_animate": ModelSpec(
        key="wan_animate",
        label="Wan 2.2 Animate-14B (reemplazo/animacion de persona)",
        repo_id="Wan-AI/Wan2.2-Animate-14B-Diffusers",
        tasks="V2V pose+face, reemplazo de persona (mode replace)",
        approx_gb=68.0,
        min_vram_gb=16,
        note="Fase 2: nucleo del flujo Video-a-Video para cambiar personas conservando el fondo.",
        phase=2,
    ),
    "wan_animate2": ModelSpec(
        key="wan_animate2",
        label="Wan 2.2 Animate-2-14B (motion transfer end-to-end)",
        repo_id="Wan-AI/Wan2.2-Animate-2-14B-Diffusers",
        tasks="V2V animacion de personaje sin extractores intermedios",
        approx_gb=68.0,
        min_vram_gb=16,
        note="Fase 2 (experimental): version 2, consume el video de driving directamente.",
        phase=2,
    ),
    "wan_vace": ModelSpec(
        key="wan_vace",
        label="Wan VACE-14B (control pose/depth/canny)",
        repo_id="Wan-AI/Wan2.1-VACE-14B-diffusers",
        tasks="V2V con condicion pose/depth/canny/trajectory",
        approx_gb=71.0,
        min_vram_gb=16,
        note="Fase 2: video de control completo + prompt para reencarnacion/restyling.",
        phase=2,
    ),
    "ltx_25": ModelSpec(
        key="ltx_25",
        label="LTX-2.5 22B distilled (video+audio nativo)",
        repo_id="Lightricks/LTX-2.5-Diffusers",
        tasks="T2V, I2V",
        approx_gb=28.0,
        min_vram_gb=16,
        note="Fase 4. FP8 + offload para 16GB.",
        phase=4,
    ),
    "preproc_pose": ModelSpec(
        key="preproc_pose",
        label="MediaPipe Pose (esqueleto cuerpo/manos/cara)",
        repo_id="",
        tasks="Preprocesador pose (V2V)",
        approx_gb=0.01,
        min_vram_gb=0,
        note="Fase 2. Se descarga solo, sin ComfyUI ni mmpose.",
        phase=2,
        direct_url="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
        filename="pose_landmarker_full.task",
    ),
    "preproc_depth": ModelSpec(
        key="preproc_depth",
        label="Depth Anything V2 (mapas de profundidad)",
        repo_id="depth-anything/Depth-Anything-V2-Small-hf",
        tasks="Preprocesador depth (V2V)",
        approx_gb=0.2,
        min_vram_gb=0,
        note="Fase 2.",
        phase=2,
    ),
    "preproc_rembg": ModelSpec(
        key="preproc_rembg",
        label="U2Net human seg (mascaras para reemplazo)",
        repo_id="",
        tasks="Preprocesador mascara (V2V replace)",
        approx_gb=0.17,
        min_vram_gb=0,
        note="Fase 2. Se descarga solo al primer uso.",
        phase=2,
    ),
}


def dir_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*")
                if f.is_file() and ".cache" not in f.parts)
    return total / 1e9


def status_of(model_dir: str, spec: ModelSpec) -> tuple[str, float]:
    """Devuelve (estado, gb_en_disco)."""
    if spec.direct_url:
        f = Path(model_dir) / "preproc" / spec.filename
        if f.exists():
            return "Descargado", f.stat().st_size / 1e9
        return "No descargado", 0.0
    if not spec.repo_id:
        return "Automatico", 0.0
    path = model_path(model_dir, spec.key)
    if not path.exists():
        return "No descargado", 0.0
    markers = list(path.glob("*.json")) or list(path.glob("*.safetensors"))
    if not markers:
        return "Parcial / vacio", dir_size_gb(path)
    return "Descargado", dir_size_gb(path)


def inventory(model_dir: str) -> list[dict]:
    rows = []
    for spec in REGISTRY.values():
        state, gb = status_of(model_dir, spec)
        rows.append(
            {
                "key": spec.key,
                "label": spec.label,
                "state": state,
                "on_disk_gb": round(gb, 2),
                "approx_gb": spec.approx_gb,
                "min_vram": spec.min_vram_gb,
                "tasks": spec.tasks,
                "note": spec.note,
                "phase": spec.phase,
            }
        )
    return rows


# ----------------------------------------------------------- progreso en vivo
_BYTES_LOCK = threading.Lock()
_BYTES_SLOTS: dict[int, list] = {}
_DL_LOCK = threading.Lock()
DOWNLOAD: dict = {"status": "idle", "key": None, "label": "", "files_done": 0,
                  "files_total": 0, "bytes_done_cum": 0, "bytes_total_cum": 0,
                  "scan_dir": "", "error": None}


class ProgressTqdm:
    """Clase tipo tqdm para huggingface_hub 1.x: registra avance en DOWNLOAD.

    snapshot_download instancia esta clase para: la barra de conteo de archivos
    (thread_map) y las dos barras agregadas de bytes ('B'): transfer y reconstruct,
    cuyas totals crecen dinamicamente via asignacion de atributo + refresh().
    """

    def __init__(self, iterable=None, total=0, initial=0, desc="", unit="",
                 unit_scale=False, unit_divisor=1024, name=None, **kwargs):
        self._iterable = iterable
        self.total = int(total or 0)
        self.n = int(initial or 0)
        self._id = id(self)
        self._bytes = (unit == "B")
        self.disable = False
        self.leave = True
        if self._bytes:
            with _BYTES_LOCK:
                _BYTES_SLOTS[self._id] = [self.total, self.n]
            with _DL_LOCK:
                DOWNLOAD["bytes_total_cum"] += max(self.total, 1)

    def update(self, step=1):
        step = int(step or 0)
        self.n += step
        if self._bytes:
            with _BYTES_LOCK:
                slot = _BYTES_SLOTS.get(self._id)
                if slot:
                    slot[1] = self.n
        else:
            with _DL_LOCK:
                DOWNLOAD["files_done"] = self.n
                if self.total:
                    DOWNLOAD["files_total"] = self.total

    def refresh(self):
        # hub actualiza .total directamente y luego llama a refresh: sincronizamos
        if self._bytes:
            with _BYTES_LOCK:
                slot = _BYTES_SLOTS.get(self._id)
                if slot:
                    slot[0] = max(slot[0], int(self.total or 0), 1)

    @property
    def format_dict(self):
        return {"n": self.n, "total": self.total, "rate": None, "elapsed": None,
                "remaining": None, "desc": "", "postfix": "", "ncols": None}

    def set_postfix_str(self, *a, **k):
        pass

    def set_description_str(self, *a, **k):
        pass

    def display(self, *a, **k):
        pass

    def clear(self, *a, **k):
        pass

    def reset(self):
        self.n = 0

    def close(self):
        if self._bytes:
            with _BYTES_LOCK:
                slot = _BYTES_SLOTS.pop(self._id, None)
            if slot:
                with _DL_LOCK:
                    DOWNLOAD["bytes_done_cum"] += slot[1]

    def __iter__(self):
        return iter(self._iterable) if self._iterable is not None else iter(())

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False

    def __getattr__(self, name):
        # set_description_str, set_postfix_str, display, etc: no-op
        return lambda *a, **k: None


def reset_progress():
    with _BYTES_LOCK:
        _BYTES_SLOTS.clear()
    with _DL_LOCK:
        DOWNLOAD.update(status="idle", key=None, label="", files_done=0,
                        files_total=0, bytes_done_cum=0, bytes_total_cum=0,
                        scan_dir="", error=None)


def start_download(model_dir: str, key: str, hf_token: str = "") -> str:
    with _DL_LOCK:
        if DOWNLOAD["status"] == "running":
            raise RuntimeError("Ya hay una descarga en curso, espera a que termine.")
    reset_progress()
    spec = REGISTRY[key]
    scan_dir = str(model_path(model_dir, spec.key)) if not spec.direct_url else ""
    with _DL_LOCK:
        DOWNLOAD.update(status="running", key=key, label=spec.label,
                        scan_dir=scan_dir)

    def _run():
        try:
            download(model_dir, key, hf_token, live=True)
            with _DL_LOCK:
                DOWNLOAD["status"] = "done"
        except Exception as exc:  # noqa: BLE001
            with _DL_LOCK:
                DOWNLOAD.update(status="error", error=str(exc)[:400])

    threading.Thread(target=_run, daemon=True).start()
    return f"Descarga iniciada: {spec.label}"


def get_progress() -> dict:
    with _BYTES_LOCK:
        live_done = sum(s[1] for s in _BYTES_SLOTS.values())
        live_total = sum(s[0] for s in _BYTES_SLOTS.values())
    with _DL_LOCK:
        status, label, key = DOWNLOAD["status"], DOWNLOAD["label"], DOWNLOAD["key"]
        files_done, files_total = DOWNLOAD["files_done"], DOWNLOAD["files_total"]
        err = DOWNLOAD["error"]
        scan_dir = DOWNLOAD.get("scan_dir") or ""
    # El conteo de bytes de XET/hub es poco fiable en agregacion: usamos lo que
    # realmente hay en disco como numerador (sin staging .cache/.incomplete),
    # y el total anunciado por hub.
    disk_bytes = 0
    if scan_dir and Path(scan_dir).exists():
        disk_bytes = sum(f.stat().st_size for f in Path(scan_dir).rglob("*")
                         if f.is_file() and ".cache" not in f.parts
                         and not f.name.endswith((".incomplete", ".metadata")))
    done = max(disk_bytes, DOWNLOAD["bytes_done_cum"] + live_done) if scan_dir else DOWNLOAD["bytes_done_cum"] + live_done
    total = DOWNLOAD["bytes_total_cum"] + live_total
    if total == 0 and key and REGISTRY[key].approx_gb:
        total = int(REGISTRY[key].approx_gb * 1e9)
    return {"status": status, "key": key, "label": label, "files_done": files_done,
            "files_total": files_total, "error": err,
            "bytes_done": done, "bytes_total": total,
            "fraction": (done / total) if total else 0.0}


def download(model_dir: str, key: str, hf_token: str = "", live: bool = False) -> str:
    """Descarga un modelo con huggingface_hub al directorio configurado."""
    spec = REGISTRY[key]
    if spec.direct_url:
        import urllib.request

        dest_dir = Path(model_dir) / "preproc"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / spec.filename
        if not dest.exists():
            urllib.request.urlretrieve(spec.direct_url, dest)  # noqa: S310
        return f"{spec.label} -> {dest}"
    from huggingface_hub import snapshot_download

    dest = model_path(model_dir, spec.key)
    dest.mkdir(parents=True, exist_ok=True)
    kwargs = {}
    if spec.allow_patterns:
        kwargs["allow_patterns"] = list(spec.allow_patterns)
    if hf_token:
        kwargs["token"] = hf_token
    if live:
        kwargs["tqdm_class"] = ProgressTqdm
    try:
        snapshot_download(
            repo_id=spec.repo_id,
            local_dir=str(dest),
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        if "gated" in str(exc).lower() or "401" in str(exc) or "403" in str(exc):
            raise RuntimeError(
                f"{spec.repo_id} requiere aceptar la licencia en HuggingFace y "
                "configurar el token HF en Ajustes."
            ) from exc
        raise
    return f"{spec.label} -> {dest}"


def delete(model_dir: str, key: str) -> str:
    spec = REGISTRY[key]
    if spec.direct_url:
        f = Path(model_dir) / "preproc" / spec.filename
        if f.exists():
            f.unlink()
            return f"Borrado {spec.label} ({f})"
        return "No habia nada que borrar."
    dest = model_path(model_dir, spec.key)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
        return f"Borrado {spec.label} ({dest})"
    return "No habia nada que borrar."


def disk_free_gb(path: str) -> float:
    drive = Path(path).drive or "/"
    return shutil.disk_usage(drive).free / 1e9
