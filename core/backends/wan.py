"""Backend Wan 2.2 TI2V-5B (diffusers).

- T2V: WanPipeline.
- I2V: WanImageToVideoPipeline (en la 5B condiciona la imagen por VAE,
  sin image_encoder CLIP; soporta image + last_image).
"""
from __future__ import annotations

MODEL_KEY = "wan_ti2v_5b"


def _prepare(pipe, resident: bool):
    import torch

    if torch.cuda.is_available() and resident:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    if hasattr(pipe, "vae") and pipe.vae is not None:
        pipe.vae.enable_tiling()
    return pipe


def load_t2v(local_dir: str, resident: bool):
    import torch
    from diffusers import WanPipeline

    pipe = WanPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16)
    return _prepare(pipe, resident)


def load_i2v(local_dir: str, resident: bool):
    import torch
    from diffusers import WanImageToVideoPipeline

    pipe = WanImageToVideoPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16)
    return _prepare(pipe, resident)


def run_t2v(pipe, prompt, negative_prompt, height, width, num_frames, steps,
            guidance, seed, step_callback=None):
    import torch

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="np",
    )
    if step_callback is not None:
        kwargs["callback_on_step_end"] = step_callback
    return pipe(**kwargs).frames[0]


def run_i2v(pipe, prompt, negative_prompt, image, height, width, num_frames, steps,
            guidance, seed, last_image=None, step_callback=None):
    import torch

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        image=image,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="np",
    )
    if last_image is not None:
        kwargs["last_image"] = last_image
    if step_callback is not None:
        kwargs["callback_on_step_end"] = step_callback
    return pipe(**kwargs).frames[0]
