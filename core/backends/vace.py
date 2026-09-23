"""Backend Wan VACE (diffusers): video de control pose/depth + prompt."""
from __future__ import annotations

MODEL_KEY = "wan_vace"


def load_vace(local_dir: str, resident: bool, quantize: bool = False):
    import torch

    if quantize:
        from diffusers import WanVACEPipeline
        from diffusers.models import WanVACETransformer3DModel

        from .. import lowvram

        transformer = lowvram.quantized_transformer(
            WanVACETransformer3DModel, local_dir,
            cache_dir=lowvram.int8_cache_dir(local_dir),
        )
        pipe = WanVACEPipeline.from_pretrained(
            local_dir, transformer=transformer, torch_dtype=torch.bfloat16
        )
        return lowvram.setup_14b(pipe)

    from diffusers import WanVACEPipeline

    pipe = WanVACEPipeline.from_pretrained(local_dir, torch_dtype=torch.bfloat16)
    if torch.cuda.is_available() and resident:
        pipe.to("cuda")
    else:
        pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    return pipe


def run_vace(
    pipe,
    control_frames,
    prompt: str = "",
    negative_prompt: str = "",
    reference_image=None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 81,
    steps: int = 30,
    guidance: float = 5.0,
    seed: int = 0,
    step_callback=None,
):
    import torch

    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        video=control_frames,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        output_type="np",
    )
    if reference_image is not None:
        kwargs["reference_images"] = [reference_image]
    if step_callback is not None:
        kwargs["callback_on_step_end"] = step_callback
    return pipe(**kwargs).frames[0]
