"""OpenAI-compatible chat client with provider fallback for the AI advisor tab.

Providers (tried in order when provider is "Auto"):

1. OpenRouter  - deepseek/deepseek-v4-flash:free (free tier, needs API key)
2. OpenCode Zen - deepseek-v4-flash-free (free tier, no API key required)
3. Groq        - llama-3.3-70b-versatile (needs API key)

All endpoints use the OpenAI chat completions format and are streamed via SSE.
"""
import json
import os

import requests

from settings import advisor_config

_ADV = advisor_config()

ADVISOR_PROVIDERS = _ADV["providers"]

AUTO_ORDER = _ADV["auto_order"]

SYSTEM_PROMPT = _ADV["system_prompt"]

STREAM_TIMEOUT = _ADV["stream_timeout"]
STREAM_TEMPERATURE = _ADV["temperature"]
STREAM_MAX_TOKENS = _ADV["max_tokens"]


def resolve_api_key(provider, extra_keys=None):
    conf = ADVISOR_PROVIDERS[provider]
    return (
        (extra_keys or {}).get(provider)
        or os.environ.get(conf["env_key"])
        or ""
    )


def stream_chat(base_url, api_key, model, messages, timeout=STREAM_TIMEOUT):
    """Stream assistant text chunks from an OpenAI-compatible endpoint."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": STREAM_TEMPERATURE,
        "max_tokens": STREAM_MAX_TOKENS,
    }
    resp = requests.post(url, headers=headers, json=payload, stream=True, timeout=timeout)
    # Providers send UTF-8 SSE. requests falls back to ISO-8859-1 when the
    # response has no charset, which mangles unicode (en dashes, etc.) into
    # mojibake ("â€“"). Force UTF-8 decoding explicitly.
    resp.encoding = "utf-8"
    if resp.status_code >= 400:
        body = resp.text[:500]
        resp.close()
        raise RuntimeError(f"{base_url} returned HTTP {resp.status_code}: {body}")
    try:
        for raw in resp.iter_lines(decode_unicode=False):
            if not raw:
                continue
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content
    finally:
        resp.close()


def call_advisor(messages, provider="Auto", extra_keys=None, system_prompt=SYSTEM_PROMPT):
    """Yield assistant text, trying providers in order until one succeeds."""
    order = AUTO_ORDER if provider == "Auto" else [provider]
    full_messages = [{"role": "system", "content": system_prompt}] + messages
    errors = []
    for name in order:
        if name not in ADVISOR_PROVIDERS:
            errors.append(f"{name}: unknown provider")
            continue
        conf = ADVISOR_PROVIDERS[name]
        api_key = resolve_api_key(name, extra_keys)
        if conf["needs_key"] and not api_key:
            errors.append(f"{name}: no API key configured")
            continue
        try:
            yield from stream_chat(conf["base_url"], api_key, conf["model"], full_messages)
            return
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    raise RuntimeError("All AI providers failed: " + " | ".join(errors))
