import os
from dataclasses import dataclass, field
from typing import Dict


# All free-tier models — provider is detected automatically from the model ID prefix
AVAILABLE_MODELS: Dict[str, Dict] = {
    # ── Google Gemini (free tier via aistudio.google.com) ──────────────────────
    "gemini-2.5-pro":        {"provider": "gemini", "label": "Gemini 2.5 Pro   [Google Free]"},
    "gemini-2.5-flash":      {"provider": "gemini", "label": "Gemini 2.5 Flash [Google Free]"},
    "gemini-2.5-flash-lite": {"provider": "gemini", "label": "Gemini 2.5 Flash-Lite [Google Free · Fastest]"},
    "gemini-1.5-flash":      {"provider": "gemini", "label": "Gemini 1.5 Flash [Google Free · Stable]"},
    # ── Google Gemma open models (served via Google AI API) ────────────────────
    "gemma-3-27b-it":        {"provider": "gemini", "label": "Gemma 3 27B      [Google Free · Open Source]"},
    "gemma-3-9b-it":         {"provider": "gemini", "label": "Gemma 3 9B       [Google Free · Open Source]"},
    # ── Groq Cloud (free tier via console.groq.com/keys) ──────────────────────
    "meta-llama/llama-4-scout-17b-16e-instruct": {"provider": "groq", "label": "LLaMA 4 Scout 17B 🖼  [Groq Free · Best Vision]"},
    "llama-3.3-70b-versatile": {"provider": "groq", "label": "LLaMA 3.3 70B        [Groq Cloud Free]"},
    "llama-3.1-8b-instant":    {"provider": "groq", "label": "LLaMA 3.1 8B Instant [Groq Cloud Free · Fastest]"},
    "mixtral-8x7b-32768":      {"provider": "groq", "label": "Mixtral 8x7B 32K     [Groq Cloud Free]"},
    "gemma2-9b-it":            {"provider": "groq", "label": "Gemma 2 9B           [Groq Cloud Free]"},
}

# gemini-1.5-flash is the safest universal default — GA (not preview), free tier,
# works with every valid Gemini API key and every version of google-generativeai.
# Startup detection will upgrade this to 2.5-flash or 2.0-flash when confirmed live.
DEFAULT_MODEL = "gemini-1.5-flash"


@dataclass
class Config:
    # API Keys — set via environment variables or HuggingFace Secrets
    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GROQ_API_KEY: str   = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    NCBI_API_KEY: str   = field(default_factory=lambda: os.getenv("NCBI_API_KEY", ""))

    # Embedding model (local, no API key required)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Retrieval settings
    MAX_PAPERS: int = 20
    TOP_K_RETRIEVAL: int = 10
    CITATION_SIMILARITY_THRESHOLD: float = 0.35

    # PubMed contact email (NCBI requires an email for E-utilities)
    PUBMED_EMAIL: str = "medreason.rag@example.com"


config = Config()
