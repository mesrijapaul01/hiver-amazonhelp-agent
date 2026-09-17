"""
Thin wrapper around the Groq API (OpenAI-compatible chat completions,
free-tier open models).

Requires env var GROQ_API_KEY. Get one free at https://console.groq.com/keys
"""
import os
import json
import time
from groq import Groq

_client = None


def get_client():
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Run `export GROQ_API_KEY=gsk_...` first "
                "(get a free key at https://console.groq.com/keys)."
            )
        _client = Groq(api_key=key)
    return _client


def call_llm(model: str, system: str, user: str, max_tokens: int = 500, retries: int = 3) -> str:
    """Single-turn call, returns text content. Retries on transient errors.

    Groq's gpt-oss models are "reasoning" models: by default they spend part of
    max_tokens on an internal reasoning pass before writing the final answer,
    which can silently eat the whole token budget and return empty content
    (finish_reason='length' with content=''). We force reasoning_effort='low'
    since these are short, simple classification/drafting tasks that don't
    need deep reasoning -- and pad max_tokens to leave headroom regardless.
    """
    client = get_client()
    last_err = None
    effective_max_tokens = max(max_tokens, 400) + 400  # headroom for reasoning overhead
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=effective_max_tokens,
                reasoning_effort="low",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            content = resp.choices[0].message.content
            if not content:
                # Ran out of tokens on the reasoning pass (even at "low" effort
                # this happens on some prompts) -- retry with a much bigger
                # budget instead of silently returning empty/None.
                resp = client.chat.completions.create(
                    model=model,
                    max_tokens=effective_max_tokens * 4,
                    reasoning_effort="low",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                content = resp.choices[0].message.content
            if not content:
                raise RuntimeError(
                    f"Model returned empty content twice in a row (finish_reason="
                    f"{resp.choices[0].finish_reason!r}) -- prompt may need to be shorter "
                    f"or max_tokens raised further."
                )
            return content
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is a take-home
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise last_err


def call_llm_json(model: str, system: str, user: str, max_tokens: int = 500) -> dict:
    """Call the LLM and parse a JSON object out of the response.
    We instruct the model to return ONLY JSON; we still defensively strip fences."""
    raw = call_llm(model, system, user, max_tokens=max_tokens)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1:
            return json.loads(cleaned[start : end + 1])
        raise
