# ControlVideoGen

Aplicacion web (Gradio) para generacion de video con modelos abiertos **sin ComfyUI**:
un solo entorno de Python para Texto-a-Video, Imagen-a-Video y Video-a-Video
(control por pose/depth, reemplazo de personas y mejora de prompts con agentes LLM).

## Modelos y backends

| Modelo | Uso | VRAM | Estado |
|---|---|---|---|
| Wan 2.2 TI2V-5B | T2V / I2V | 6–16 GB (offload) | ✅ probado end-to-end |
| Wan 2.2 Animate-14B | V2V: reemplazo/animacion de persona (pose+car+cara, conserva fondo) | 16 GB (offload) | código listo, requiere descarga (~33 GB) |
| Wan VACE-14B (2.1) | V2V: control pose / depth / canny + prompt | 16 GB (offload) | código listo, requiere descarga (~33 GB) |
| LTX-2.5 22B distilled | T2V / I2V con audio nativo | 16 GB (offload, fp8) | requiere aceptar licencia en HF + token |
| MiniMax H3 (API oficial) | T2V / I2V / referencias 768P–2K | n/a (nube) | requiere API key pay-as-you-go |

> **Nota MiniMax**: verificado con plan de tokens (sk-cp, Plus): **chat M3 y `image-01`
> SÍ funcionan** (se usan como agente de prompts y motor de swap). El **video H3 NO**
> está cubierto por el token plan (error 2013); requiere Creditos pay-as-you-go de la
> plataforma. Y H3 local necesita ~75 GB de RAM del sistema, no viable en 32 GB.

## Instalacion (un solo environment)

```powershell
.\setup.ps1    # crea .venv, instala PyTorch CUDA (cu128) y dependencias
.\run.ps1      # lanza la app en http://127.0.0.1:7860
```

## Primeros pasos

1. Pestana **Modelos y Ajustes** → elige la carpeta de modelos (ej. `F:\ControlVideoGen\models`)
   y pulsa **Guardar ruta**. Todo (pesos, caches HF, rembg, MediaPipe) va ahi; el disco C no se satura.
2. Descarga **Wan 2.2 TI2V-5B** (~32 GB) y genera desde **Texto a Video**.
3. Para V2V: descarga **Wan Animate** o **VACE** y abre la pestana **Video a Video**.

## Perfil de VRAM

El selector Auto detecta tu GPU. Perfiles de 6–16 GB usan **offload por modulos**
(muy importante en Windows: mantener todo residente con spill de driver degrada
~100x la velocidad). Referencia en RTX 5070 Ti 16 GB: T2V 480p/49f ≈ 2.1 s/paso,
I2V 480p/25f ≈ 10 min total.

## Flujo Video a Video (pose to video)

1. Sube el video → la app extrae **pose (MediaPipe)**, **depth (Depth Anything V2)**,
   **mascara de persona (rembg)** y **recortes de cara** — con preview del esqueleto.
2. **Camino A — Wan Animate `replace`**: imagen de la persona nueva + pose + cara +
   mascara → se reemplaza SOLO la persona, conservando fondo y audio del video.
3. **Camino A lite — `animate`**: la persona nueva anima sobre la pose (fondo generado).
4. **Camino B — VACE**: video de esqueleto/depth completo + prompt → reencarnacion o
   re-estilizado total (otro personaje, estilo, entorno).
5. Los modelos 14B generan en segmentos de ~77 frames con condicion temporal del
   segmento previo; al final se re-ensambla y se mezcla el audio original.

### Swap rapido con Nano Banana / GPT-Image / image-01

En V2V, el panel "Swap rapido" toma un frame del video + una foto de la persona
nueva y genera la imagen de referencia con el motor elegido (Gemini
`gemini-2.5-flash-image`, OpenAI `gpt-image-1` o MiniMax `image-01`), que luego
alimenta a Wan Animate.

### MiniMax en la nube (Token Plan)

Modo **Video a Video → "MiniMax Hailuo-2.3 (nube, TOKEN PLAN)"**: la app replica el
flujo del CLI de MiniMax, que si funciona con un plan de tokens (a diferencia de H3,
que exige Creditos de cuenta):

1. **Preflight de media** — `probe_media` detecta duracion/codec con ffmpeg; si el
   video no cumple (VP9, o fuera de 2-15 s) `prepare_reference_media` transcode a
   H.264 CRF20 + AAC igual que el CLI.
2. **Auto-prompt de motion** — muestrea frames del video y se los pasa a **M3** (elegible
   para el plan de tokens) que devuelve una descripcion de coreografia beat-by-beat,
   neutra en genero.
3. **Imagen-a-video first-frame** — sube la imagen como `first_frame_image` (el plan
   cubre Hailuo-2.3) con el prompt I2VA; la app envuelve el texto en el formato
   `"...at 0.00 seconds, <Picture 1> is fully referenced"`, hace polling y descarga.
   Limites: 6 o 10 s @ 768P, sin audio nativo.

Boton en V2V: **🕺 Auto-prompt motion con M3** (rellena el prompt desde el video).
Nota: `MiniMax H3` (r2va, 15 s, audio sincronizado) sigue necesitando Creditos de
cuenta (HTTP 402/2013) — el plan de tokens no lo cubre.

### Agente de prompts

Boton "Mejorar prompt" en cada pestana, con proveedor seleccionable
(DeepSeek / OpenAI / Anthropic / Gemini). Configura la API key en Ajustes.

## Estructura

```
app.py                  # UI Gradio (T2V, I2V, V2V, Modelos y Ajustes)
core/
  config.py             # settings.json + rutas (HF_HOME etc. antes de importar)
  model_manager.py      # registro, descarga, inventario, borrado de modelos
  engine.py             # perfiles de VRAM, cache de pipelines, generacion T2V/I2V/V2V
  preprocess.py         # pose (MediaPipe), depth, mascaras (rembg), cara
  prompt_agent.py       # mejorador multi-proveedor
  character_swap.py     # Nano Banana (Gemini) para generar referencias de persona
  backends/
    wan.py              # Wan 2.2 TI2V-5B (T2V/I2V)
    wan_animate.py      # Wan Animate (animate/replace)
    vace.py             # Wan VACE (pose/depth control)
    ltx.py              # LTX-2.5 (gated HF)
    minimax_api.py      # MiniMax H3 via API
outputs/                # videos generados (auto-creado)
```
