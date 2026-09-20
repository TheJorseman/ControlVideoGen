"""Backend LTX-2.5 22B distilled (diffusers, bf16 + offload para 16GB).

El repo Lightricks/LTX-2.5-Diffusers es gated: hay que aceptar la licencia en
Hugging Face y configurar el token en Ajustes.
"""
from __future__ import annotations

MODEL_KEY = "ltx_25"

FRAME_CHOICES = [49, 65, 81, 97, 113, 121]  # %8 == 1


def load_ltx(local_dir: str, resident: bool):
    import torch
    from diffusers import LTX2ImageToVideoPipeline

    pipe = LTX2ImageToVideoPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available() and resident:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def run_ltx(pipe, prompt: str, negative_prompt: str = "", image=None,
            height: int = 512, width: int = 768, num_frames: int = 81,
            steps: int = 30, guidance: float = 3.0, seed: int = 0, fps: float = 24.0,
            step_callback=None):
    import torch

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        height=height,
        width=width,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    if image is not None:
        kwargs["image"] = image
    if step_callback is not None:
        kwargs["callback_on_step_end"] = step_callback
    result = pipe(**kwargs)
    return result.frames[0]
