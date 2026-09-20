"""Agente mejorador de prompts: DeepSeek / OpenAI / Anthropic / Gemini via HTTP."""
from __future__ import annotations

SYSTEM = (
    "You are an expert prompt engineer for text-to-video diffusion models "
    "(Wan 2.2, LTX-2.5). Expand the user's prompt into a vivid, detailed video prompt: "
    "subject, action, environment, camera movement, lighting, style and mood. "
    "Keep it under 120 words, one paragraph, no bullets, no quotes. "
    "Reply ONLY with the improved prompt."
)

PROVIDERS = {
    "deepseek": {
        "label": "DeepSeek",
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
    },
    "openai": {
        "label": "OpenAI",
        "url": "https://api.openai.com/v1/chat/completions",
        "model": "gpt-4o-mini",
    },
    "anthropic": {
        "label": "Anthropic",
        "url": "https://api.anthropic.com/v1/messages",
        "model": "claude-sonnet-4-5",
    },
    "gemini": {
        "label": "Gemini",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        "model": "gemini-2.5-flash",
    },
}


def enhance_prompt(prompt: str, provider: str, api_key: str, model: str = "") -> str:
    if provider not in PROVIDERS:
        raise RuntimeError(f"Proveedor desconocido: {provider}")
    if not api_key:
        raise RuntimeError(f"Falta la API key de {PROVIDERS[provider]['label']} (Ajustes).")
    cfg = PROVIDERS[provider]
    model = model or cfg["model"]
    import httpx

    if provider == "anthropic":
        resp = httpx.post(
            cfg["url"],
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": model, "max_tokens": 400, "system": SYSTEM,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()
    if provider == "gemini":
        resp = httpx.post(
            cfg["url"].format(model=model),
            params={"key": api_key},
            json={"systemInstruction": {"parts": [{"text": SYSTEM}]},
                  "contents": [{"parts": [{"text": prompt}]}]},
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    # deepseek y openai son OpenAI-compatible
    resp = httpx.post(
        cfg["url"],
        headers={"authorization": f"Bearer {api_key}", "content-type": "application/json"},
        json={"model": model, "max_tokens": 400, "temperature": 0.7,
              "messages": [{"role": "system", "content": SYSTEM},
                           {"role": "user", "content": prompt}]},
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()
