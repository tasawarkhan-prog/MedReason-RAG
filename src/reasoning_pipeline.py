from __future__ import annotations
import json
import re
import requests
from typing import Dict, List


_DIAGNOSTICIAN_PROMPT = """You are an expert clinical AI assistant for EDUCATIONAL PURPOSES ONLY.

DISCLAIMER: This tool is for research/educational demonstration. All outputs must NOT be used for actual clinical decisions.

---
PATIENT CASE:
{case_text}

EXTRACTED ENTITIES:
Symptoms: {symptoms}
Labs: {labs}
Known diagnoses: {known_diagnoses}
Medications: {medications}

RETRIEVED PUBMED EVIDENCE (use these to ground your reasoning):
{evidence_text}

---
TASK: Generate exactly 5 differential diagnoses ordered from most to least likely.

Return ONLY valid JSON, no markdown fences, matching this exact schema:
{{
  "diagnoses": [
    {{
      "condition": "Full condition name",
      "likelihood": "High|Medium|Low",
      "supporting_features": ["feature from the case", "..."],
      "reasoning": "2-3 sentence clinical reasoning citing specific PMIDs where relevant",
      "citations": [
        {{"pmid": "12345678", "relevance": "one-sentence relevance"}}
      ],
      "confirmatory_tests": ["test1", "test2"],
      "confidence_score": 0.85
    }}
  ]
}}"""

_DEVILS_ADVOCATE_PROMPT = """You are a critical-appraisal AI reviewing a differential diagnosis list for EDUCATIONAL PURPOSES ONLY.

Original case summary: {case_summary}

Proposed diagnoses:
{diagnoses_text}

Challenge each diagnosis. Return ONLY valid JSON:
{{
  "challenges": [
    {{
      "condition": "condition name",
      "contradicting_evidence": "why the proposed diagnosis could be wrong",
      "alternative_explanation": "what else fits the picture",
      "missed_critical": "dangerous condition that must be ruled out first (if any)"
    }}
  ],
  "critical_alert": "Most urgent life-threatening condition to exclude immediately, or empty string"
}}"""

# Groq Cloud API endpoint (OpenAI-compatible)
_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _detect_provider(model_id: str) -> str:
    """Look up provider from AVAILABLE_MODELS; infer from name prefix for dynamic models."""
    from config import AVAILABLE_MODELS
    if model_id in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_id].get("provider", "gemini")
    # Infer provider for dynamically discovered models not in the static registry.
    # Groq serves: LLaMA, Mixtral, Gemma2, DeepSeek, Qwen, Mistral, and its own
    # compound/routing models (groq/...) plus OpenAI models relayed through Groq (openai/...).
    mid = model_id.lower()
    groq_prefixes = (
        "llama", "meta-llama/", "mixtral", "gemma2", "whisper",
        "groq/", "openai/", "deepseek", "qwen", "mistral",
    )
    if any(mid.startswith(p) for p in groq_prefixes):
        return "groq"
    return "gemini"


class ReasoningPipeline:
    """
    Multi-agent reasoning (Diagnostician + Devil's Advocate).
    Supports all free-tier models: Gemini, Gemma (via Google AI), and Groq Cloud
    (LLaMA, Mixtral, Gemma2). Provider is auto-detected from AVAILABLE_MODELS registry.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self._api_key = api_key
        self._model_id = model
        self._provider = _detect_provider(model)

        if self._provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._gemini_model = genai.GenerativeModel(model)

    # ------------------------------------------------------------------
    # Provider-dispatched call
    # ------------------------------------------------------------------

    def _call(self, prompt: str) -> str:
        if self._provider == "groq":
            return self._call_groq(prompt)
        return self._call_gemini(prompt)

    def _call_gemini(self, prompt: str) -> str:
        response = self._gemini_model.generate_content(prompt)
        return response.text

    def _call_groq(self, prompt: str) -> str:
        response = requests.post(
            _GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 4096,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_json(self, text: str) -> dict:
        cleaned = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        m = re.search(r"\{[\s\S]*\}", cleaned)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        return {}

    def _format_evidence(self, papers: List[Dict]) -> str:
        lines = []
        for p in papers[:10]:
            title = p.get("title", "")[:100]
            abstract = p.get("abstract", "")[:400]
            lines.append(f"[PMID:{p.get('pmid','?')}] {title}\n  {abstract}...")
        return "\n\n".join(lines) if lines else "No evidence retrieved."

    def _fallback_diagnoses(self, raw_text: str) -> List[Dict]:
        return [{
            "condition": "Unable to parse structured output — see raw text",
            "likelihood": "Unknown",
            "supporting_features": [],
            "reasoning": raw_text[:800],
            "citations": [],
            "confirmatory_tests": [],
            "confidence_score": 0.0,
        }]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_differential_diagnosis(
        self,
        case_text: str,
        entities: Dict,
        evidence_papers: List[Dict],
    ) -> Dict:
        evidence_text = self._format_evidence(evidence_papers)

        # Agent 1 — Diagnostician
        diag_prompt = _DIAGNOSTICIAN_PROMPT.format(
            case_text=case_text[:1200],
            symptoms=", ".join(entities.get("symptoms", [])) or "none",
            labs=", ".join(entities.get("labs", [])) or "none",
            known_diagnoses=", ".join(entities.get("diagnoses", [])) or "none",
            medications=", ".join(entities.get("medications", [])) or "none",
            evidence_text=evidence_text,
        )
        diag_raw = self._call(diag_prompt)
        diag_data = self._parse_json(diag_raw)
        diagnoses: List[Dict] = diag_data.get("diagnoses", [])
        if not diagnoses:
            diagnoses = self._fallback_diagnoses(diag_raw)

        # Agent 2 — Devil's Advocate
        diagnoses_text = "\n".join(
            f"{i+1}. {d.get('condition','?')} ({d.get('likelihood','?')}): "
            f"{d.get('reasoning','')[:200]}"
            for i, d in enumerate(diagnoses)
        )
        da_prompt = _DEVILS_ADVOCATE_PROMPT.format(
            case_summary=case_text[:400],
            diagnoses_text=diagnoses_text,
        )
        da_raw = self._call(da_prompt)
        da_data = self._parse_json(da_raw)

        return {
            "diagnoses": diagnoses,
            "devils_advocate": da_data,
            "model_used": self._model_id,
            "provider": self._provider,
            "clinical_disclaimer": (
                "EDUCATIONAL USE ONLY — NOT FOR CLINICAL DECISIONS. "
                "Always consult a qualified healthcare professional."
            ),
        }
