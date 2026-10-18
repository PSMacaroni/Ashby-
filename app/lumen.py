"""
Plain-language slide summaries from NCSA Lumen (an OpenAI-compatible service).

Lumen only rewords what the analysis already says: it gets the analysis sentences (no images, no
patient data) and must not add numbers, diagnoses or treatment advice. We check that every number
in its answer also appears in what we sent; if not, or if Lumen is off or slow, the app shows the
template summary instead (explain.plain_summary).

Set up: in the project's .env file (never commit it, never paste keys in chat):
    LUMEN_API_KEY=...   LUMEN_BASE_URL=...   LUMEN_MODEL=...
"""
import os
import re
import threading

from common import ROOT

TIMEOUT_S = 20
SYSTEM = (
    "You rewrite pathology slide-analysis findings for a pathologist. Use only the facts and numbers "
    "given. Do not add numbers, diagnoses, staging conclusions or treatment advice. Keep words like "
    "'consistent with', 'check' and 'possible'. Write one short paragraph, at most 90 words, plain and calm."
)

_cache = {}
_lock = threading.Lock()


def settings():
    """(api key, base url, model) from .env; empty strings if not set."""
    try:
        from dotenv import dotenv_values
        env = dotenv_values(ROOT / ".env")
    except Exception:
        env = {}
    get = lambda k: (os.getenv(k) or env.get(k) or "").strip()
    return get("LUMEN_API_KEY"), get("LUMEN_BASE_URL"), get("LUMEN_MODEL")


def available():
    key, url, model = settings()
    return bool(key and url and model)


def numbers(text):
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def summarize(facts):
    """
    facts: the template paragraph. Returns (text, source) with source "Lumen" or "template";
    never raises. Answers are remembered, so each slide is sent once.
    """
    if not available():
        return facts, "template"
    with _lock:
        if facts in _cache:
            return _cache[facts]
    key, url, model = settings()
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url=url, timeout=TIMEOUT_S, max_retries=0)
        reply = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=220,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": facts}])
        text = (reply.choices[0].message.content or "").strip()
    except Exception as e:
        return facts, f"template (Lumen error: {type(e).__name__})"
    extra = numbers(text) - numbers(facts)
    result = (text, "Lumen") if text and not extra else (facts, "template (Lumen added numbers, so not shown)")
    with _lock:
        _cache[facts] = result
    return result
