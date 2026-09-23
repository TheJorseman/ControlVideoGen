"""Prueba de conexion MiniMax. Uso:

    python test_minimax.py sk-cp-XXXX            # con la key como argumento
    set MINIMAX_KEY=sk-cp-XXXX && python test_minimax.py

Prueba, de mas barato a mas caro:
  1) ListModels / endpoint gratuito de validacion de key
  2) image-01 text-to-image (una imagen)
  3) chat completions (para el prompt agent / ref2va)
La prueba de video (H3 ref2va) NO se lanza por defecto porque consume saldo;
usa --video para forzarla.
"""
from __future__ import annotations

import json
import os
import sys

import httpx

from core import config
from core.backends import minimax_api as mm


def _key() -> str:
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        return sys.argv[1].strip()
    env = os.environ.get("MINIMAX_KEY", "").strip()
    if env:
        return env
    saved = config.load_settings()["api_keys"].get("minimax", "").strip()
    if saved:
        return saved
    print("No se encontró API key. Pásala como argumento o en MINIMAX_KEY.")
    sys.exit(2)


def main() -> None:
    key = _key()
    host = mm._base(os.environ.get("MINIMAX_HOST", "https://api.minimax.io"))
    kind = "sk-cp (plan de tokens)" if key.startswith("sk-cp") else "JWT/plataforma"
    print(f"Host: {host}")
    print(f"Key: {key[:8]}...{key[-4:]}  (tipo detectado: {kind})")
    print("=" * 60)

    results = {}

    # 1) Validacion barata: list models via endpoint de chat compatible-anthropic
    print("\n[1] Verificando la key contra el endpoint Anthropic-compatible (chat)...")
    try:
        r = httpx.post(
            f"{host}/anthropic/v1/messages",
            headers={"authorization": f"Bearer {key}",
                     "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": "MiniMax-M3", "max_tokens": 16,
                  "messages": [{"role": "user", "content": "Say OK"}]},
            timeout=60,
        )
        print(f"    HTTP {r.status_code}")
        body = r.json()
        if r.status_code == 200:
            txt = (body.get("content") or [{}])[0].get("text", "").strip()
            print(f"    OK respuesta: {txt!r}")
            results["chat_anthropic"] = "ok"
        else:
            print("    " + json.dumps(body)[:300])
            results["chat_anthropic"] = f"http {r.status_code}"
    except Exception as exc:  # noqa: BLE001
        print(f"    fallo: {exc}")
        results["chat_anthropic"] = str(exc)[:120]

    # 2) image-01 text-to-image (barato, valida el endpoint de imagen)
    print("\n[2] Probando image-01 text-to-image (gasta unas centésimas)...")
    try:
        data = mm.generate_image(key, "a red circle on white background, minimal",
                                 reference_path=None, aspect_ratio="1:1", host=host)
        print(f"    OK: {len(data)} bytes de imagen PNG/JPEG")
        out = os.path.join("outputs", "test_minimax_image.png")
        os.makedirs("outputs", exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(data)
        print(f"    guardado en {out}")
        results["image01"] = "ok"
    except Exception as exc:  # noqa: BLE001
        print(f"    fallo: {exc}")
        results["image01"] = str(exc)[:160]

    # 3) video H3: SOLO con --video (consume mas saldo)
    if "--video" in sys.argv:
        print("\n[3] Probando video H3 text-to-video (5s, 768P)... consume saldo.")
        try:
            def cb(_):
                print("    esperando...")
            url = mm.generate(key, "a cat waving hello, simple", kind="t2v",
                              duration=5, resolution="768P", ratio="16:9",
                              poll_cb=cb, host=host)
            print("    URL:", url[:80])
            results["video_h3"] = "ok"
        except Exception as exc:  # noqa: BLE001
            print(f"    fallo: {exc}")
            results["video_h3"] = str(exc)[:160]
    else:
        print("\n[3] Video H3 NO probado (usa --video para forzarlo).")
        results["video_h3"] = "skipped"

    print("\n" + "=" * 60)
    print("RESUMEN:", json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
