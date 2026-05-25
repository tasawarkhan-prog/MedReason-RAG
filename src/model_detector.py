"""
Dynamically detect available free models from Gemini and Groq live APIs.
Falls back to the static AVAILABLE_MODELS list if the API call fails.
"""
from __future__ import annotations
from typing import List, Tuple
import re
import requests

# Gemini live API returns versioned IDs like "gemini-2.5-flash-preview-05-20".
# We normalise these to the stable aliases our AVAILABLE_MODELS registry uses,
# so the model selected in the UI is the same ID passed to generate_content().
_GEMINI_NORMALISE = [
    # Order matters — more-specific patterns first
    (r"^gemini-2\.5-flash-lite",  "gemini-2.5-flash-lite"),
    (r"^gemini-2\.5-flash",       "gemini-2.5-flash"),
    (r"^gemini-2\.5-pro",         "gemini-2.5-pro"),
    (r"^gemini-2\.0-flash",       "gemini-2.0-flash"),
    (r"^gemini-1\.5-flash",       "gemini-1.5-flash"),
    (r"^gemini-1\.5-pro",         "gemini-1.5-pro"),
    (r"^gemma-3-27b",             "gemma-3-27b-it"),
    (r"^gemma-3-9b",              "gemma-3-9b-it"),
]

def _normalize_gemini_id(mid: str) -> str:
    for pattern, stable in _GEMINI_NORMALISE:
        if re.match(pattern, mid):
            return stable
    return mid

# Preferred order — most reliable free model first, then newest
_GEMINI_PREFERENCE = [
    "gemini-1.5-flash",       # GA, universally available — always the safe default
    "gemini-2.5-flash",       # Best free model when available
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-pro",
    "gemma-3-27b-it",
    "gemma-3-9b-it",
]
_GROQ_PREFERENCE = [
    "llama-3.3-70b-versatile",                      # Best text model — default Groq choice
    "llama-3.1-70b-versatile",
    "llama3-70b-8192",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "meta-llama/llama-4-scout-17b-16e-instruct",     # Vision model — available but not default
]

# Model ID substrings that indicate non-chat / non-reasoning Groq models to exclude
_GROQ_SKIP_TAGS = (
    "whisper", "tts", "guard",        # Audio + safety classifiers
    "groq/",                           # Internal routing models (groq/compound etc.)
    "openai/",                         # OpenAI models routed via Groq — unreliable
    "canopy",                          # canopylabs/orpheus — TTS
    "playai",                          # PlayAI TTS
    "allam",                           # Arabic-only model
)

# Labels for models that may appear live but aren't in our static list
_EXTRA_LABELS: dict = {
    "gemini-2.0-flash":         "Gemini 2.0 Flash [Google Free · Auto-detected]",
    "gemini-exp-1206":          "Gemini Exp 1206 [Google Free · Auto-detected]",
    "llama3-70b-8192":          "LLaMA 3 70B [Groq Free · Auto-detected]",
    "llama-3.1-70b-versatile":  "LLaMA 3.1 70B [Groq Free · Auto-detected]",
}


def _label(model_id: str) -> str:
    from config import AVAILABLE_MODELS
    return (
        AVAILABLE_MODELS.get(model_id, {}).get("label")
        or _EXTRA_LABELS.get(model_id)
        or f"{model_id} [Auto-detected]"
    )


def _static_gemini() -> List[Tuple[str, str]]:
    from config import AVAILABLE_MODELS
    return [
        (_label(mid), mid)
        for mid, info in AVAILABLE_MODELS.items()
        if info["provider"] == "gemini"
    ]


def _static_groq() -> List[Tuple[str, str]]:
    from config import AVAILABLE_MODELS
    return [
        (_label(mid), mid)
        for mid, info in AVAILABLE_MODELS.items()
        if info["provider"] == "groq"
    ]


def detect_gemini_models(api_key: str) -> List[Tuple[str, str]]:
    """Return [(label, model_id)] for available Gemini generateContent models."""
    live_ids: set = set()
    got_live = False
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        # Normalise versioned IDs (e.g. gemini-2.5-flash-preview-05-20 → gemini-2.5-flash)
        # so the selected model ID is valid when passed to GenerativeModel().
        live_ids = {
            _normalize_gemini_id(m.name.split("/")[-1])
            for m in genai.list_models()
            if "generateContent" in (getattr(m, "supported_generation_methods", None) or [])
        }
        got_live = bool(live_ids)
    except Exception:
        pass

    from config import AVAILABLE_MODELS
    static_ids = {mid for mid, info in AVAILABLE_MODELS.items() if info["provider"] == "gemini"}

    if got_live:
        # Trust the live API — only show models confirmed for this key.
        # Prevents static-only models (e.g. gemini-2.5-pro) from being shown when
        # the key doesn't have access, which would cause 404 errors during analysis.
        usable = live_ids
    else:
        # Live call failed — fall back to known-good static models.
        usable = static_ids

    all_known = set(_GEMINI_PREFERENCE) | usable
    ordered   = _GEMINI_PREFERENCE + sorted(all_known - set(_GEMINI_PREFERENCE))

    results = [(_label(mid), mid) for mid in ordered if mid in usable]
    return results or _static_gemini()


def detect_groq_models(api_key: str) -> List[Tuple[str, str]]:
    """Return [(label, model_id)] for available Groq chat-completion models."""
    live_ids: set = set()
    got_live = False
    try:
        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        live_ids = {
            m["id"] for m in resp.json().get("data", [])
            if not any(tag in m["id"].lower() for tag in _GROQ_SKIP_TAGS)
        }
        got_live = bool(live_ids)
    except Exception:
        pass

    from config import AVAILABLE_MODELS
    static_ids = {mid for mid, info in AVAILABLE_MODELS.items() if info["provider"] == "groq"}

    if got_live:
        # Trust the live API — only show models that Groq confirmed for this account.
        # This prevents static models (e.g. LLaMA 4 Scout) from being listed when the
        # account doesn't actually have access to them.
        usable = live_ids
    else:
        # Live API unreachable or returned nothing — fall back to known-good static list
        usable = static_ids

    all_known = set(_GROQ_PREFERENCE) | usable
    ordered   = _GROQ_PREFERENCE + sorted(all_known - set(_GROQ_PREFERENCE))

    results = [(_label(mid), mid) for mid in ordered if mid in usable]
    return results or _static_groq()


def best_available_model(
    gemini_key: str = "",
    groq_key: str = "",
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Return (best_model_id, all_choices) based on which API keys are provided.
    Gemini choices come first (preferred), then Groq.
    Falls back to the full static list when no keys are given.
    """
    from config import AVAILABLE_MODELS, DEFAULT_MODEL

    choices: List[Tuple[str, str]] = []

    if gemini_key.strip():
        choices.extend(detect_gemini_models(gemini_key.strip()))
    if groq_key.strip():
        choices.extend(detect_groq_models(groq_key.strip()))

    if not choices:
        all_static = [(info["label"], mid) for mid, info in AVAILABLE_MODELS.items()]
        return DEFAULT_MODEL, all_static

    return choices[0][1], choices
