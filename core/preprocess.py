"""Preprocesadores para Video-a-Video: pose, depth, mascaras y cara.

Todo se ejecuta sobre frames decodificados del video de entrada y devuelve
listas de PIL.Image listas para los pipelines de condicionamiento.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

_CACHE: dict = {}


# ------------------------------------------------------------------ video IO
def load_video(path: str, max_frames: int | None = None, size: tuple | None = None):
    """Devuelve (frames np uint8 [N,H,W,3], fps) usando OpenCV para leer."""
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No se pudo leer el video: {path}")
    out = np.stack(frames)
    if size:
        out = np.stack([cv2.resize(f, size) for f in out])
    return out, float(fps)


def save_video(frames, path: str, fps: float = 24.0, audio_from: str | None = None):
    """frames: np [N,H,W,3] o lista de PIL. Escribe H.264 con imageio-ffmpeg."""
    import imageio.v3 as iio

    arr = np.stack([_to_np(f) for f in frames])
    iio.imwrite(path, arr, fps=fps)
    if audio_from:
        _mux_audio(path, audio_from)
    return path


def _to_np(frame):
    if isinstance(frame, np.ndarray):
        return frame
    return np.asarray(frame.convert("RGB"))


def _mux_audio(video_path: str, audio_src: str):
    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    out = video_path + ".tmp.mp4"
    cmd = [
        ffmpeg, "-y", "-i", video_path, "-i", audio_src,
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-shortest", out,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        Path(out).replace(video_path)
    except (FileNotFoundError, subprocess.CalledProcessError):
        Path(out).unlink(missing_ok=True)


# ------------------------------------------------------------------ pose
# Conexiones del esqueleto (indices landmark de MediaPipe, estilo OpenPose).
_POSE_CONNECTIONS = [
    ((0, 11), (255, 0, 0)), ((0, 12), (255, 0, 0)),          # cabeza
    ((11, 12), (0, 255, 0)), ((23, 24), (0, 255, 0)),        # torso
    ((11, 23), (0, 255, 0)), ((12, 24), (0, 255, 0)),
    ((11, 13), (255, 255, 0)), ((13, 15), (255, 255, 0)),    # brazo izq
    ((12, 14), (0, 255, 255)), ((14, 16), (0, 255, 255)),    # brazo der
    ((15, 17), (255, 128, 0)), ((15, 18), (255, 128, 0)), ((15, 19), (255, 128, 0)),
    ((16, 20), (128, 0, 255)), ((16, 21), (128, 0, 255)), ((16, 22), (128, 0, 255)),
    ((23, 25), (0, 128, 255)), ((25, 27), (0, 128, 255)),    # pierna izq
    ((24, 26), (255, 0, 128)), ((26, 28), (255, 0, 128)),    # pierna der
    ((27, 29), (128, 128, 128)), ((29, 31), (128, 128, 128)),
    ((28, 30), (128, 128, 128)), ((30, 32), (128, 128, 128)),
]
_FACE_LANDMARKS = list(range(1, 11))


def _ensure_pose_model(model_dir: str) -> Path:
    from .model_manager import REGISTRY

    spec = REGISTRY["preproc_pose"]
    dest = Path(model_dir) / "preproc" / spec.filename
    if not dest.exists():
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(spec.direct_url, dest)  # noqa: S310
    return dest


def _get_landmarker(model_dir: str):
    if "mp_pose" not in _CACHE:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(_ensure_pose_model(model_dir))
            ),
            num_poses=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        _CACHE["mp_pose"] = vision.PoseLandmarker.create_from_options(opts)
    return _CACHE["mp_pose"]


def _detect_landmarks(landmarker, frame_np):
    import mediapipe as mp

    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_np)
    result = landmarker.detect(img)
    return result.pose_landmarks[0] if result.pose_landmarks else None


def extract_pose(frames_np, model_dir: str = "models", min_visibility: float = 0.4):
    """Esqueletos estilo OpenPose sobre fondo negro via MediaPipe Tasks."""
    import cv2

    landmarker = _get_landmarker(model_dir)
    outs = []
    for f in frames_np:
        h, w, _ = f.shape
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        pts = _detect_landmarks(landmarker, f)
        if pts:
            for (a, b), color in _POSE_CONNECTIONS:
                pa, pb = pts[a], pts[b]
                if min(pa.visibility, pb.visibility) < min_visibility:
                    continue
                p1 = (int(pa.x * w), int(pa.y * h))
                p2 = (int(pb.x * w), int(pb.y * h))
                cv2.line(canvas, p1, p2, color, max(2, h // 200))
            for i in _FACE_LANDMARKS:
                if pts[i].visibility >= min_visibility:
                    cv2.circle(canvas, (int(pts[i].x * w), int(pts[i].y * h)),
                               max(2, h // 250), (255, 255, 255), -1)
        outs.append(_np_to_pil(canvas))
    return outs


# ------------------------------------------------------------------ depth
def extract_depth(frames_np, model_id: str = "depth-anything/Depth-Anything-V2-Small-hf"):
    if "depth_pipe" not in _CACHE or _CACHE.get("depth_id") != model_id:
        from transformers import pipeline

        _CACHE["depth_pipe"] = pipeline(model=model_id, device=_device())
        _CACHE["depth_id"] = model_id
    pipe = _CACHE["depth_pipe"]
    import PIL.Image

    outs = []
    for f in frames_np:
        result = pipe(PIL.Image.fromarray(f))
        pred = result["predicted_depth"]
        d = np.asarray(pred, dtype=np.float32)
        d = (d - d.min()) / (d.max() - d.min() + 1e-8) * 255
        outs.append(_np_to_pil(d.astype(np.uint8)))
    return outs


# ------------------------------------------------------------------ mask
def extract_person_masks(frames_np, invert: bool = False):
    """Mascara de persona por frame via rembg. Blanco = generar, negro = conservar."""
    if "rembg" not in _CACHE:
        from rembg import new_session

        _CACHE["rembg"] = new_session("u2net_human_seg")
    from rembg import remove

    session = _CACHE["rembg"]
    outs = []
    for f in frames_np:
        alpha = remove(f, only_alpha=True, session=session)
        a = np.asarray(alpha)
        mask = (a > 127).astype(np.uint8) * 255
        if invert:
            mask = 255 - mask
        outs.append(_np_to_pil(mask))
    return outs


# ------------------------------------------------------------------ face
def extract_face_crops(frames_np, model_dir: str = "models", mask_frames=None,
                       size: int = 512):
    """Recorte de cara usando landmarks de MediaPipe (nariz/hombros) con fallback a
    bbox de la mascara o region superior central."""
    import cv2

    landmarker = _get_landmarker(model_dir)
    outs = []
    for i, f in enumerate(frames_np):
        h, w, _ = f.shape
        pts = _detect_landmarks(landmarker, f)
        box = None
        if pts and min(pts[0].visibility, pts[11].visibility, pts[12].visibility) > 0.4:
            nose = np.array([pts[0].x * w, pts[0].y * h])
            sh_l = np.array([pts[11].x * w, pts[11].y * h])
            sh_r = np.array([pts[12].x * w, pts[12].y * h])
            span = float(np.linalg.norm(sh_l - sh_r))
            side = max(span * 1.15, h * 0.15)
            cx, cy = nose[0], (nose[1] * 2 + (sh_l[1] + sh_r[1]) / 2) / 3
            x0, y0 = int(cx - side / 2), int(cy - side / 2)
            box = (x0, y0, side, side)
        elif mask_frames is not None:
            m = np.asarray(mask_frames[i].convert("L"))
            ys, xs = np.where(m > 127)
            if len(xs):
                box = (int(xs.min()), int(ys.min()),
                       int(xs.max() - xs.min()), int(ys.max() - ys.min()))
        if box is None:
            cx, cy = w // 2, h // 4
            side = min(h, w) // 3
            box = (cx - side // 2, cy - side // 2, side, side)
        x, y, bw, bh = (int(v) for v in box)
        pad = int(0.18 * max(bw, bh))
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + bw + pad, w), min(y + bh + pad, h)
        x1, y1 = max(x1, x0 + 8), max(y1, y0 + 8)
        crop = cv2.resize(f[y0:y1, x0:x1], (size, size))
        outs.append(_np_to_pil(crop))
    return outs


# ------------------------------------------------------------------ utils
def _device():
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _np_to_pil(arr):
    import PIL.Image

    return PIL.Image.fromarray(arr)


def frames_to_pil(frames_np):
    return [_np_to_pil(f) for f in frames_np]


def pil_to_frames(pil_list, size=None):
    import cv2

    arr = []
    for im in pil_list:
        a = np.asarray(im.convert("RGB"))
        if size and (a.shape[1], a.shape[0]) != size:
            a = cv2.resize(a, size)
        arr.append(a)
    return np.stack(arr)
