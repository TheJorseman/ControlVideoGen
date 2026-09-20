"""Backend Wan 2.2 Animate (diffusers): animacion y reemplazo de personaje."""
from __future__ import annotations

MODEL_KEY = "wan_animate"


def load_animate(local_dir: str, resident: bool):
    import torch
    from diffusers import WanAnimatePipeline

    pipe = WanAnimatePipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available() and resident:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def run_animate(
    pipe,
    image,
    pose_frames,
    face_frames,
    background_frames=None,
    mask_frames=None,
    prompt: str = "",
    negative_prompt: str = "",
    mode: str = "animate",
    height: int = 720,
    width: int = 1280,
    segment_frame_length: int = 77,
    steps: int = 20,
    guidance: float = 1.0,
    seed: int = 0,
    step_callback=None,
):
    import torch

    kwargs = dict(
        image=image,
        pose_video=pose_frames,
        face_video=face_frames,
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        height=height,
        width=width,
        segment_frame_length=segment_frame_length,
        num_inference_steps=steps,
        guidance_scale=guidance,
        mode=mode,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="np",
    )
    if mode == "replace":
        kwargs["background_video"] = background_frames
        kwargs["mask_video"] = mask_frames
    if step_callback is not None:
        kwargs["callback_on_step_end"] = step_callback
    return pipe(**kwargs).frames[0]
