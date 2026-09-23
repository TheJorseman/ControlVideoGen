"""Utilidades de memoria: cuantizacion INT8 (torchao) + streaming de bloques.

Para modelos 14B de Wan (~64 GB fp32 en disco / ~28 GB bf16) en maquinas con
16 GB de VRAM y poca RAM:

- El transformer se cuantiza a INT8 (~14.5 GB) y, tras la primera conversion,
  se GUARDA en un cache local para que las cargas siguientes lean el INT8
  directamente (sin tocar los 64 GB fp32 ni picos de RAM altos).
- El group-offload usa modo NO-streaming sin paginas pineadas: en Windows el
  pinned memory + streams provoca cudaErrorAlreadyMapped y suma RAM bloqueada.
  Sin pin, los pesos viven como memoria normal y el pico baja ~8-10 GB.
"""
from __future__ import annotations

from pathlib import Path

import torch


def quantized_transformer(cls, local_dir: str, subfolder: str = "transformer",
                          cache_dir: str | None = None):
    """Carga INT8 usando cache cuantizado si existe; si no, convierte y cachea."""
    from diffusers import TorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    cache = Path(cache_dir) if cache_dir else None
    if cache is not None and (cache / "config.json").exists():
        try:
            return cls.from_pretrained(str(cache), torch_dtype=torch.bfloat16)
        except Exception:  # noqa: BLE001
            pass  # cache corrupto: reconstruir

    model = cls.from_pretrained(
        local_dir,
        subfolder=subfolder,
        torch_dtype=torch.bfloat16,
        quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2)),
        low_cpu_mem_usage=False,
    )
    if cache is not None:
        try:
            cache.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(cache))
        except Exception:  # noqa: BLE001
            pass  # sin espacio: seguimos sin cache
    return model


def int8_cache_dir(local_dir: str) -> str:
    return str(Path(local_dir).parent / (Path(local_dir).name + "_int8"))


def quantize_module(module):
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    module.requires_grad_(False)
    quantize_(module, Int8WeightOnlyConfig(version=2))


def stream_pipeline(pipe, device: str = "cuda"):
    """Bloques del transformer y hojas del text encoder viajan CPU->GPU bajo
    demanda (RAM pageable, sin pin, sin stream). Encoders residuales a GPU."""
    from diffusers.hooks import apply_group_offloading

    onload = torch.device(device)
    off = torch.device("cpu")

    pipe.transformer.requires_grad_(False)
    try:
        apply_group_offloading(
            pipe.transformer, offload_type="block_level", num_blocks_per_group=1,
            onload_device=onload, offload_device=off,
            use_stream=False, non_blocking=False,
        )
    except Exception:  # noqa: BLE001
        apply_group_offloading(pipe.transformer, offload_type="leaf_level",
                               onload_device=onload, offload_device=off,
                               use_stream=False)
        return pipe

    # modulos fuera de los bloques (motion/face encoder, embeddings): a GPU.
    # Se usa Module.to() por hijo: los tensores cuantizados (subclasses de
    # torchao) no admiten reasignar param.data entre dispositivos.
    for name, child in pipe.transformer.named_children():
        if name == "blocks":
            continue
        child.to(onload)

    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.requires_grad_(False)
        base = getattr(pipe.text_encoder, "model", pipe.text_encoder)
        apply_group_offloading(base, offload_type="leaf_level",
                               onload_device=onload, offload_device=off,
                               use_stream=False)
    if getattr(pipe, "image_encoder", None) is not None:
        pipe.image_encoder.to(device)
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.to(device)
        try:
            pipe.vae.enable_tiling()
        except Exception:  # noqa: BLE001
            pass
    return pipe


def setup_14b(pipe):
    """Ruta completa: cuantiza text encoder y configura streaming por bloques."""
    if torch.cuda.is_available():
        quantize_module(pipe.text_encoder)
        stream_pipeline(pipe, device="cuda")
    else:
        pipe.to("cpu")
    return pipe
