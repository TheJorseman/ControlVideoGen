"""ControlVideoGen — app Gradio para generacion de video con modelos abiertos.

Uso:  python app.py   (abre http://127.0.0.1:7860)
"""
from __future__ import annotations

import logging
import random

import gradio as gr

from core import config, engine, model_manager
from core import character_swap
from core.model_manager import REGISTRY


class _ResetNoiseFilter(logging.Filter):
    """Silencia el ConnectionResetError del proactor de Windows al cerrar el
    navegador conexiones SSE de la cola de Gradio (inofensivo)."""

    def filter(self, record):
        exc = record.exc_info[1] if isinstance(record.exc_info, tuple) and len(record.exc_info) == 3 else None
        noisy = "_call_connection_lost" in str(record.getMessage())
        return not (noisy and isinstance(exc, ConnectionError))


logging.getLogger("asyncio").addFilter(_ResetNoiseFilter())

SETTINGS = config.load_settings()
config.apply_env(SETTINGS["model_dir"])

GPU = engine.detect_gpu()

PROFILE_CHOICES = [
    ("Auto (detectar GPU)", "auto"),
    ("6 GB", "6gb"),
    ("8 GB", "8gb"),
    ("12 GB", "12gb"),
    ("16 GB", "16gb"),
    ("24 GB", "24gb"),
]


# ---------------------------------------------------------------- helpers UI
def fmt_bytes_table() -> list[list]:
    rows = []
    for item in model_manager.inventory(SETTINGS["model_dir"]):
        rows.append(
            [
                item["label"],
                item["state"],
                f"{item['on_disk_gb']:.2f}",
                f"{item['approx_gb']:.1f}",
                f"{item['min_vram']:.0f}",
                f"Fase {item['phase']}",
            ]
        )
    return rows


def header_md() -> str:
    prof = engine.resolve_profile(SETTINGS)
    return (
        f"**GPU detectada:** {GPU['name']} ({GPU['vram_gb']} GB, CUDA {GPU['torch_cuda']}) | "
        f"**Perfil VRAM:** {SETTINGS['vram_profile']} -> {prof['vram_gb']} GB "
        f"(max {prof['max_resolution']}, {prof['max_frames']} frames) | {engine.disk_report(SETTINGS)}"
    )


def save_model_dir(new_dir: str):
    SETTINGS["model_dir"] = new_dir.strip()
    config.save_settings(SETTINGS)
    config.apply_env(new_dir)
    return header_md(), fmt_bytes_table()


def save_profile(p: str):
    SETTINGS["vram_profile"] = p
    config.save_settings(SETTINGS)
    return header_md()


def download_selected(key: str):
    if not key:
        return "Selecciona un modelo."
    try:
        return model_manager.start_download(
            SETTINGS["model_dir"], key, SETTINGS.get("hf_token", "")
        )
    except Exception as exc:  # noqa: BLE001
        return f"Error: {exc}"


def _bar_html(fraction: float, text: str) -> str:
    pct = max(0.0, min(1.0, fraction)) * 100
    return (
        f'<div style="width:100%;background:#e5e7eb;border-radius:8px;height:26px;'
        f'position:relative;overflow:hidden;margin:6px 0">'
        f'<div style="width:{pct:.1f}%;background:linear-gradient(90deg,#2563eb,#16a34a);'
        f'height:100%;transition:width 1s"></div>'
        f'<div style="position:absolute;inset:0;display:flex;align-items:center;'
        f'justify-content:center;font-size:13px;font-weight:600">{text}</div></div>'
    )


def poll_download():
    """Timer: actualiza barra, texto y tabla segun estado de descarga."""
    p = model_manager.get_progress()
    if p["status"] == "running":
        txt = (f"{p['files_done']}/{p['files_total'] or '?'} archivos · "
               f"{p['bytes_done'] / 1e9:.1f}/{p['bytes_total'] / 1e9:.1f} GB · "
               f"{p['fraction'] * 100:.0f}% · {p['label']}")
        return _bar_html(p["fraction"], txt), fmt_bytes_table(), gr.Button(interactive=False)
    if p["status"] == "done":
        return (_bar_html(1.0, f"✅ Descarga completada: {p['label']}"),
                fmt_bytes_table(), gr.Button(interactive=True))
    if p["status"] == "error":
        return (_bar_html(0.0, "❌ Error: ve el mensaje bajo la tabla"),
                fmt_bytes_table(), gr.Button(interactive=True))
    return (
        _bar_html(0.0, "Sin descarga en curso"),
        fmt_bytes_table(), gr.Button(interactive=True),
    )


def delete_selected(key: str, confirm: bool):
    if not key:
        return fmt_bytes_table(), "Selecciona un modelo."
    if not confirm:
        return fmt_bytes_table(), "Marca la casilla de confirmacion primero."
    msg = model_manager.delete(SETTINGS["model_dir"], key)
    return fmt_bytes_table(), msg


def save_keys(hf_token, deepseek, openai, anthropic, gemini, minimax, minimax_host):
    SETTINGS["hf_token"] = hf_token
    SETTINGS["minimax_host"] = (minimax_host or "https://api.minimax.io").strip()
    SETTINGS["api_keys"] = {
        "deepseek": deepseek,
        "openai": openai,
        "anthropic": anthropic,
        "gemini": gemini,
        "minimax": minimax,
    }
    config.save_settings(SETTINGS)
    return ("Claves guardadas. MiniMax ahora puede usarse como motor de swap (image-01) "
            "y como modo V2V en la nube (H3 ref2va). Host: " + SETTINGS["minimax_host"])


def poll_gen():
    """Timer: devuelve (barra HTML, texto) segun engine.PROGRESS."""
    s = engine.progress_state()
    if s["done"]:
        stage = s["stage"] or "listo"
        return _bar_html(s.get("fraction", 1.0), f"✅ {stage}"), f"Estado: {stage}"
    if s["pre_total"]:
        frac = s["pre_step"] / max(s["pre_total"], 1)
        txt = f"{s['stage']}: {s['pre_step']}/{s['pre_total']} frames ({frac * 100:.0f}%)"
        return _bar_html(frac, txt), txt
    if s["total_steps"]:
        frac = s["fraction"]
        txt = (f"{s['stage']} — segmento {s['segment']}/{s['total_segments']} · "
               f"paso {s['step']}/{s['steps_per_seg']} · {frac * 100:.0f}% · "
               f"{s['avg_step']:.0f}s/paso · ETA {int(s['eta'] // 60)}m{int(s['eta'] % 60):02d}s")
        return _bar_html(frac, txt), txt
    txt = s["stage"] or "..."
    return _bar_html(0.0, txt), txt


def resolve_seed(seed: int) -> int:
    return random.randint(0, 2**31 - 1) if seed is None or seed < 0 else int(seed)


def frame_choices() -> list[int]:
    m = engine.max_frames(SETTINGS)
    return [f for f in range(17, min(m, 121) + 1, 4)]


# ---------------------------------------------------------------- generacion
def do_generate(kind: str, prompt, negative, resolution, frames, steps, guidance, seed,
                backend="wan_5b", image=None, last_image=None):
    try:
        out = engine.generate(
            settings=SETTINGS,
            kind=kind,
            prompt=prompt,
            negative_prompt=negative,
            resolution=resolution,
            num_frames=frames,
            steps=steps,
            guidance=guidance,
            seed=resolve_seed(seed),
            image_path=image,
            last_image_path=last_image,
            backend=backend,
        )
        return out, f"Video generado: {out}"
    except Exception as exc:  # noqa: BLE001
        return None, f"Error: {exc}"


def do_extract(video_path, mode, resolution, frames):
    if not video_path:
        return None, "Sube un video primero."
    try:
        conds = engine.prepare_conditions(SETTINGS, video_path, mode, resolution, int(frames))
        return conds["preview"], (
            f"Condiciones extraidas: {len(conds['frames'])} frames · "
            f"{conds['width']}x{conds['height']} · "
            f"{len(conds['frames']) / conds['fps']:.1f}s a {conds['fps']:.0f}fps. Puedes generar."
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"Error: {exc}"


def do_describe_motion(video_path, duration):
    if not video_path:
        return None, "Sube primero el video de referencia."
    try:
        text = engine.describe_motion(SETTINGS, video_path, target_duration=int(duration))
        return text, f"Auto-prompt generado por M3 ({len(text.split())} palabras, con timestamps)."
    except Exception as exc:  # noqa: BLE001
        return None, f"Error M3: {exc}"


def do_sync_audio(gen_video, src_video):
    if not gen_video or not src_video:
        return None, "Necesitas el video generado Y el video fuente (con el audio)."
    try:
        path, note = engine.sync_generated_audio(gen_video, src_video, SETTINGS)
        return path, f" {note} → {path}"
    except Exception as exc:  # noqa: BLE001
        return None, f"Error: {exc}"


def do_v2v(video_path, mode, character, prompt, negative, resolution, frames,
           steps, guidance, seed, start_ref, duration):
    if not video_path:
        return None, None, "Sube un video primero."
    try:
        out, preview = engine.generate_v2v(
            SETTINGS, mode, video_path, prompt, negative, resolution, int(frames),
            int(steps), float(guidance), resolve_seed(seed), character,
            start_reference=bool(start_ref), duration_s=int(duration),
        )
        return preview, out, f"Video generado: {out}"
    except Exception as exc:  # noqa: BLE001
        return None, None, f"Error: {exc}"


def do_enhance(prompt: str, provider: str) -> str:
    from core import prompt_agent

    if not prompt.strip():
        return prompt
    try:
        return prompt_agent.enhance_prompt(
            prompt, provider, SETTINGS["api_keys"].get(provider, ""),
            host=SETTINGS.get("minimax_host", ""),
        )
    except Exception as exc:  # noqa: BLE001
        return prompt


def do_nano_swap(video_path, person_image, engine_label: str, model_label: str,
                 scene_prompt: str = ""):
    from core import character_swap

    if not video_path or not person_image:
        return None, "Sube el video de referencia Y la foto de la persona nueva."
    try:
        if engine_label == "GPT-Image-1 (OpenAI)":
            out = character_swap.swap_with_openai(
                video_path, person_image, SETTINGS["api_keys"].get("openai", "")
            )
            return out, f"Referencia generada con GPT-Image-1: {out}"
        if engine_label == "MiniMax image-01":
            out = character_swap.swap_with_minimax(
                video_path, person_image, SETTINGS["api_keys"].get("minimax", ""),
                scene_prompt=scene_prompt, host=SETTINGS.get("minimax_host", ""),
            )
            return out, f"Referencia generada con MiniMax image-01: {out}"
        model = character_swap.NANO_MODELS.get(model_label, "gemini-2.5-flash-image")
        out = character_swap.generate_character_reference(
            video_path, person_image, SETTINGS["api_keys"].get("gemini", ""), model
        )
        return out, f"Referencia generada con {model_label}: {out}"
    except Exception as exc:  # noqa: BLE001
        return None, f"Error swap ({engine_label}): {exc}"


with gr.Blocks(title="ControlVideoGen") as demo:
    gr.Markdown("# 🎬 ControlVideoGen\nGeneracion de video con modelos abiertos (Wan 2.2, LTX-2.5) sin ComfyUI.")
    header = gr.Markdown(header_md())

    with gr.Tabs():
        # ------------------------------------------------------------ T2V
        with gr.Tab("Texto a Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    t2v_backend = gr.Dropdown(engine.BACKEND_CHOICES, value="wan_5b",
                                              label="Modelo / backend")
                    t2v_prompt = gr.Textbox(label="Prompt", lines=3,
                                            placeholder="A cinematic drone shot of a red sports car driving along a coastal road at sunset...")
                    with gr.Row():
                        t2v_enhance = gr.Button("✨ Mejorar prompt con agente LLM", size="sm")
                        t2v_provider = gr.Dropdown(
                            [("DeepSeek", "deepseek"), ("OpenAI", "openai"),
                             ("Anthropic", "anthropic"), ("Gemini", "gemini"), ("MiniMax M3 (plan tokens)", "minimax")],
                            value="deepseek", label="Proveedor", scale=2)
                    t2v_negative = gr.Textbox(label="Prompt negativo", lines=1,
                                              value="worst quality, blurry, distorted")
                    with gr.Accordion("Avanzado", open=False):
                        t2v_resolution = gr.Dropdown(choices=engine.resolution_options(SETTINGS),
                                                     label="Resolucion", value="832x480 (16:9)")
                        t2v_frames = gr.Dropdown(choices=frame_choices(), label="Frames (24 fps)",
                                                 value=81 if 81 in frame_choices() else frame_choices()[-1])
                        t2v_steps = gr.Slider(8, 50, value=30, step=1, label="Pasos de deduccion")
                        t2v_guidance = gr.Slider(1.0, 10.0, value=1.0, step=0.5,
                                                 label="Guidance (recomendado 1.0 en Wan 2.2-5B)")
                        t2v_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = aleatorio)")
                    with gr.Row():
                        t2v_btn = gr.Button("Generar", variant="primary")
                        cancel_btn = gr.Button("Cancelar")
                    t2v_bar = gr.HTML(_bar_html(0.0, "Sin generacion"))
                    t2v_status = gr.Markdown()
                    t2v_timer = gr.Timer(2.0, active=True)
                    t2v_timer.tick(fn=poll_gen, outputs=[t2v_bar, t2v_status])
                with gr.Column(scale=2):
                    t2v_out = gr.Video(label="Resultado", format="mp4")

        # ------------------------------------------------------------ I2V
        with gr.Tab("Imagen a Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    i2v_backend = gr.Dropdown(engine.BACKEND_CHOICES, value="wan_5b",
                                              label="Modelo / backend")
                    i2v_image = gr.Image(type="filepath", label="Imagen de inicio")
                    i2v_last = gr.Image(type="filepath", label="Ultima imagen (opcional)")
                    i2v_prompt = gr.Textbox(label="Prompt (describe el movimiento/escena)", lines=3)
                    with gr.Row():
                        i2v_enhance = gr.Button("✨ Mejorar prompt con agente LLM", size="sm")
                        i2v_provider = gr.Dropdown(
                            [("DeepSeek", "deepseek"), ("OpenAI", "openai"),
                             ("Anthropic", "anthropic"), ("Gemini", "gemini"), ("MiniMax M3 (plan tokens)", "minimax")],
                            value="deepseek", label="Proveedor", scale=2)
                    i2v_negative = gr.Textbox(label="Prompt negativo", lines=1,
                                              value="worst quality, blurry, distorted")
                    with gr.Accordion("Avanzado", open=False):
                        i2v_resolution = gr.Dropdown(choices=engine.resolution_options(SETTINGS),
                                                     label="Resolucion", value="832x480 (16:9)")
                        i2v_frames = gr.Dropdown(choices=frame_choices(), label="Frames (24 fps)",
                                                 value=81 if 81 in frame_choices() else frame_choices()[-1])
                        i2v_steps = gr.Slider(8, 50, value=30, step=1, label="Pasos de deduccion")
                        i2v_guidance = gr.Slider(1.0, 10.0, value=1.0, step=0.5, label="Guidance")
                        i2v_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = aleatorio)")
                    i2v_btn = gr.Button("Generar", variant="primary")
                    i2v_bar = gr.HTML(_bar_html(0.0, "Sin generacion"))
                    i2v_status = gr.Markdown()
                    i2v_timer = gr.Timer(2.0, active=True)
                    i2v_timer.tick(fn=poll_gen, outputs=[i2v_bar, i2v_status])
                with gr.Column(scale=2):
                    i2v_out = gr.Video(label="Resultado", format="mp4")

        # ------------------------------------------------------------ V2V
        with gr.Tab("Video a Video"):
            gr.Markdown(
                "**Flujo:** sube el video → elige modo → (imagen de la persona nueva si aplica) → prompt → generar. "
                "La app extrae pose/depth/mascara automaticamente y muestra el video de condicion antes de generar. "
                "Nota: modos con Wan 14B usan offload a RAM en 16GB (lento pero funciona)."
            )
            with gr.Row():
                with gr.Column(scale=2):
                    v2v_video = gr.Video(label="Video de referencia", sources="upload")
                    v2v_mode = gr.Dropdown(choices=engine.v2v_modes(), value="animate_replace",
                                           label="Modo de control")
                    v2v_duration = gr.Radio(choices=[("6 s", 6), ("10 s", 10)], value=10,
                                            label="Duracion para modos nube (Hailuo-2.3; H3 usa Max frames/24)")
                    v2v_character = gr.Image(type="filepath",
                                             label="Imagen de la persona nueva (Animate / referencia en VACE)",
                                             interactive=True)
                    with gr.Accordion("⚡ Swap rapido: generar referencia con Nano Banana / GPT-Image", open=True):
                        gr.Markdown("Usa un frame del video + una foto de la persona nueva para generar la "
                                    "imagen de referencia automaticamente (API key de Gemini o de OpenAI).")
                        nano_engine = gr.Radio(choices=list(character_swap.SWAP_ENGINES.keys()),
                                               value="Nano Banana (Gemini)", label="Motor de swap")
                        with gr.Row():
                            nano_person = gr.Image(type="filepath", label="Foto de la persona nueva")
                            nano_model = gr.Dropdown(choices=list(character_swap.NANO_MODELS.keys()),
                                                     value="Nano Banana (rapido)",
                                                     label="Modelo Gemini (ignorado con OpenAI)")
                        nano_btn = gr.Button("Generar imagen de referencia")
                        nano_msg = gr.Markdown()
                    with gr.Accordion("🔊 Sync de audio post-generacion (opc. 4: recompensar offsets)", open=False):
                        gr.Markdown("El audio de Hailuo/H2.3 no calza con la coreografia inventada: "
                                    "carga aqui el clip generado y el video fuente y la app correlaciona "
                                    "la energia de movimiento frame a frame para cortar/retrasar el audio.")
                        with gr.Row():
                            sync_gen = gr.Video(label="Video generado")
                            sync_src = gr.Video(label="Video fuente (con el audio)")
                        sync_btn = gr.Button("Sincronizar audio")
                        sync_msg = gr.Markdown()
                        sync_out = gr.Video(label="Resultado sincronizado")
                    v2v_prompt = gr.Textbox(label="Prompt", lines=3,
                                            placeholder="A woman in a red dress dancing... / describe the new look")
                    with gr.Row():
                        v2v_enhance = gr.Button("✨ Mejorar prompt con agente LLM", size="sm")
                        v2v_m3 = gr.Button("🕺 Auto-prompt motion con M3 (del video)", size="sm")
                        v2v_provider = gr.Dropdown(
                            [("DeepSeek", "deepseek"), ("OpenAI", "openai"),
                             ("Anthropic", "anthropic"), ("Gemini", "gemini"),
                             ("MiniMax M3 (plan tokens)", "minimax")],
                            value="deepseek", label="Proveedor", scale=2)
                    v2v_negative = gr.Textbox(label="Prompt negativo", lines=1,
                                              value="worst quality, blurry, distorted")
                    with gr.Accordion("Avanzado", open=False):
                        v2v_resolution = gr.Dropdown(
                            choices=[engine.AUTO_RESOLUTION] + engine.resolution_options(SETTINGS),
                            label="Resolucion (Auto respeta la orientacion del video, sin deformar)",
                            value=engine.AUTO_RESOLUTION)
                        v2v_frames = gr.Slider(17, 161, value=81, step=4, label="Max frames (se ajusta a 4N+1)")
                        v2v_steps = gr.Slider(8, 50, value=24, step=1,
                                              label="Pasos por segmento (24+ mejora la identidad)")
                        v2v_guidance = gr.Slider(1.0, 10.0, value=1.0, step=0.5, label="Guidance")
                        v2v_seed = gr.Number(value=-1, precision=0, label="Seed (-1 = aleatorio)")
                        v2v_start_ref = gr.Checkbox(
                            label="Empezar con la imagen de referencia (crossfade 0.5s al inicio)",
                            value=False)
                    with gr.Row():
                        v2v_extract_btn = gr.Button("Solo extraer condiciones (preview)")
                        v2v_btn = gr.Button("Generar V2V", variant="primary")
                    v2v_bar = gr.HTML(_bar_html(0.0, "Sin generacion"))
                    v2v_status = gr.Markdown()
                    v2v_timer = gr.Timer(3.0, active=True)
                    v2v_timer.tick(fn=poll_gen, outputs=[v2v_bar, v2v_status])
                with gr.Column(scale=2):
                    v2v_cond_preview = gr.Video(label="Video de condicion extraido")
                    v2v_out = gr.Video(label="Resultado", format="mp4")

        # ------------------------------------------------------------ MODELOS
        with gr.Tab("Modelos y Ajustes"):
            gr.Markdown("### 💾 Almacen de modelos (evita saturar el disco C)")
            with gr.Row():
                dir_box = gr.Textbox(label="Carpeta de modelos", value=SETTINGS["model_dir"], scale=4)
                dir_save = gr.Button("Guardar ruta", scale=1)
            gr.Markdown(
                "Todos los pesos, caches de HuggingFace y preprocesadores se guardan ahi. "
                "La carpeta `outputs/` (videos generados) queda junto a la app."
            )

            gr.Markdown("### ⚡ Perfil de VRAM")
            profile_box = gr.Dropdown(PROFILE_CHOICES, value=SETTINGS["vram_profile"],
                                      label="Perfil VRAM objetivo", info="Controla offload a RAM, resolucion maxima y frames.")
            with gr.Row():
                unload_btn = gr.Button("Liberar GPU (descargar modelo)")
                refresh_btn = gr.Button("Refrescar inventario")

            gr.Markdown("### 📦 Modelos")
            table = gr.Dataframe(
                headers=["Modelo", "Estado", "En disco (GB)", "Descarga aprox. (GB)", "VRAM minima", "Fase"],
                value=fmt_bytes_table(),
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                sel_model = gr.Dropdown([(s.label, s.key) for s in REGISTRY.values()],
                                        label="Modelo seleccionado", value="wan_ti2v_5b")
                with gr.Column():
                    dl_btn = gr.Button("⬇ Descargar", variant="primary")
                    del_btn = gr.Button("🗑 Borrar")
                confirm_del = gr.Checkbox(label="Confirmar borrado", value=False)
            dl_bar = gr.HTML(_bar_html(0.0, "Sin descarga en curso"))
            dl_timer = gr.Timer(2.0, active=True)
            model_msg = gr.Markdown()

            gr.Markdown("### 🔑 API keys y token HuggingFace")
            with gr.Row():
                hf_token_box = gr.Textbox(label="Token HuggingFace (necesario para repos gated como LTX-2.5)",
                                          type="password", value=SETTINGS.get("hf_token", ""))
                k_deepseek = gr.Textbox(label="DeepSeek", type="password", value=SETTINGS["api_keys"]["deepseek"])
                k_openai = gr.Textbox(label="OpenAI", type="password", value=SETTINGS["api_keys"]["openai"])
                k_anthropic = gr.Textbox(label="Anthropic", type="password", value=SETTINGS["api_keys"]["anthropic"])
            with gr.Row():
                k_gemini = gr.Textbox(label="Gemini (Nano Banana)", type="password", value=SETTINGS["api_keys"]["gemini"])
                k_minimax = gr.Textbox(label="MiniMax (H3 por API)", type="password", value=SETTINGS["api_keys"]["minimax"])
                k_minimax_host = gr.Textbox(label="MiniMax host",
                                            value=SETTINGS.get("minimax_host", "https://api.minimax.io"),
                                            info="Usa https://api.minimaxi.com si tu cuenta es de la version china")
                keys_save = gr.Button("Guardar keys")
            keys_msg = gr.Markdown()

    # ------------------------------------------------------------ eventos
    t2v_btn.click(fn=lambda *a: do_generate("t2v", *a),
                  inputs=[t2v_prompt, t2v_negative, t2v_resolution, t2v_frames,
                          t2v_steps, t2v_guidance, t2v_seed, t2v_backend],
                  outputs=[t2v_out, t2v_status])
    i2v_btn.click(fn=lambda *a: do_generate("i2v", *a),
                  inputs=[i2v_prompt, i2v_negative, i2v_resolution, i2v_frames,
                          i2v_steps, i2v_guidance, i2v_seed, i2v_backend,
                          i2v_image, i2v_last],
                  outputs=[i2v_out, i2v_status])
    cancel_btn.click(fn=engine.interrupt, outputs=t2v_status)
    unload_btn.click(fn=engine.unload, outputs=model_msg)
    t2v_enhance.click(fn=do_enhance, inputs=[t2v_prompt, t2v_provider], outputs=t2v_prompt)
    i2v_enhance.click(fn=do_enhance, inputs=[i2v_prompt, i2v_provider], outputs=i2v_prompt)
    v2v_enhance.click(fn=do_enhance, inputs=[v2v_prompt, v2v_provider], outputs=v2v_prompt)
    nano_btn.click(fn=do_nano_swap,
                   inputs=[v2v_video, nano_person, nano_engine, nano_model, v2v_prompt],
                   outputs=[v2v_character, nano_msg])
    v2v_extract_btn.click(fn=do_extract,
                          inputs=[v2v_video, v2v_mode, v2v_resolution, v2v_frames],
                          outputs=[v2v_cond_preview, v2v_status])
    v2v_btn.click(fn=do_v2v,
                  inputs=[v2v_video, v2v_mode, v2v_character, v2v_prompt, v2v_negative,
                          v2v_resolution, v2v_frames, v2v_steps, v2v_guidance, v2v_seed,
                          v2v_start_ref, v2v_duration],
                  outputs=[v2v_cond_preview, v2v_out, v2v_status])
    v2v_m3.click(fn=do_describe_motion, inputs=[v2v_video, v2v_duration],
                 outputs=[v2v_prompt, v2v_status])
    sync_btn.click(fn=do_sync_audio, inputs=[sync_gen, sync_src],
                   outputs=[sync_out, sync_msg])

    dir_save.click(fn=save_model_dir, inputs=dir_box, outputs=[header, table])
    profile_box.change(fn=save_profile, inputs=profile_box, outputs=header)
    refresh_btn.click(fn=fmt_bytes_table, outputs=table)
    dl_btn.click(fn=download_selected, inputs=sel_model, outputs=model_msg)
    dl_timer.tick(fn=poll_download, outputs=[dl_bar, table, dl_btn])
    del_btn.click(fn=delete_selected, inputs=[sel_model, confirm_del], outputs=[table, model_msg])
    keys_save.click(fn=save_keys,
                    inputs=[hf_token_box, k_deepseek, k_openai, k_anthropic, k_gemini,
                            k_minimax, k_minimax_host],
                    outputs=keys_msg)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(inbrowser=True, theme=gr.themes.Soft())
