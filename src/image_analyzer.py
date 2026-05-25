"""
Analyze uploaded medical images (X-ray, MRI, CT, photo) and PDF reports.
Returns a clinical text description that feeds into the main RAG pipeline.
Free APIs only: Gemini Vision (primary) + Groq LLaMA-4 Scout Vision (fallback).
"""
from __future__ import annotations
import base64
import requests
from pathlib import Path
from typing import Tuple

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
_PDF_EXT    = ".pdf"

_VISION_PROMPT = """You are an expert radiologist and medical AI assistant (for EDUCATIONAL USE ONLY).
Analyze this medical content carefully and produce a structured clinical case description.

Include ALL of the following that are visible or readable:
- Type of content (X-ray, MRI, CT scan, ECG, clinical photograph, lab report, pathology slide, etc.)
- Patient demographics if visible (age, sex, body part)
- Key findings and abnormalities — describe location, size, density/signal, character
- Relevant normal findings
- Anatomical regions and laterality
- Severity or urgency indicators
- Likely associated symptoms
- Suggested differential diagnoses based on the image/document alone

Format the output as a detailed clinical case description (3–5 paragraphs) that a physician could use
to look up relevant evidence. Use proper medical terminology. Be specific and comprehensive.

DISCLAIMER: This is for research and educational purposes only, not clinical use."""


def _mime(path: str) -> str:
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
        ".webp": "image/webp", ".bmp": "image/png", ".gif": "image/gif",
        ".tiff": "image/jpeg", ".tif": "image/jpeg",
    }.get(Path(path).suffix.lower(), "image/jpeg")


def _gemini_vision(file_path: str, api_key: str) -> str:
    """Use Gemini 1.5 Flash (free multimodal) to analyze a medical image or PDF page."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    with open(file_path, "rb") as f:
        data = f.read()
    ext = Path(file_path).suffix.lower()
    mime = "application/pdf" if ext == _PDF_EXT else _mime(file_path)
    response = model.generate_content([
        _VISION_PROMPT,
        {"mime_type": mime, "data": data},
    ])
    return response.text.strip()


def _groq_vision(file_path: str, api_key: str) -> str:
    """Use Groq LLaMA-4 Scout Vision (free) to analyze a medical image."""
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    data_url = f"data:{_mime(file_path)};base64,{b64}"
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": _VISION_PROMPT},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            "temperature": 0.3,
            "max_tokens": 1200,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _extract_pdf_text(file_path: str) -> str:
    """Extract text from a PDF using pypdf (no API key needed)."""
    text = ""
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(p.strip() for p in pages if p.strip())
    except Exception:
        try:
            import PyPDF2
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(p.strip() for p in pages if p.strip())
        except Exception:
            pass
    return text[:5000]


def describe_upload(
    file_path: str,
    gemini_api_key: str = "",
    groq_api_key: str = "",
) -> Tuple[str, str]:
    """
    Analyze an uploaded file and return (clinical_description, status_message).

    Strategy:
    - PDF: extract text first (free, no API); if empty, try Gemini Vision
    - Image: try Gemini Vision, then Groq Vision
    """
    ext = Path(file_path).suffix.lower()

    # ── PDF ──────────────────────────────────────────────────────────────────
    if ext == _PDF_EXT:
        text = _extract_pdf_text(file_path)
        if text.strip():
            return text, "PDF text extracted. Review and click Analyse."
        if gemini_api_key.strip():
            try:
                desc = _gemini_vision(file_path, gemini_api_key.strip())
                return desc, "PDF analyzed with Gemini Vision. Review description and click Analyse."
            except Exception as e:
                return "", f"PDF analysis failed: {e}"
        return "", "Could not extract text from this PDF. Add a Gemini API key to enable AI analysis."

    # ── Image ─────────────────────────────────────────────────────────────────
    if ext in _IMAGE_EXTS:
        if gemini_api_key.strip():
            try:
                desc = _gemini_vision(file_path, gemini_api_key.strip())
                return desc, "Image analyzed with Gemini Vision. Review description and click Analyse."
            except Exception:
                pass
        if groq_api_key.strip():
            try:
                desc = _groq_vision(file_path, groq_api_key.strip())
                return desc, "Image analyzed with Groq Vision. Review description and click Analyse."
            except Exception as e:
                return "", f"Image analysis failed: {e}"
        return "", "Image analysis requires a Gemini or Groq API key. Please add one above."

    return "", f"Unsupported file type '{ext}'. Please upload a JPG, PNG, or PDF."
