"""
MedReason-RAG — HuggingFace Spaces entry point.
Modern, professional medical-AI UI design.
"""
import os
import traceback
from typing import Generator

import gradio as gr

from config import config, AVAILABLE_MODELS, DEFAULT_MODEL
from src.pubmed_retriever import PubMedRetriever
from src.medical_ner import MedicalNER
from src.query_expander import QueryExpander
from src.embeddings import EmbeddingEngine
from src.vector_store import VectorStore
from src.reasoning_pipeline import ReasoningPipeline, _detect_provider
from src.citation_verifier import CitationVerifier
from src.drug_checker import DrugChecker
from src.utils import format_diagnosis_output
from src.image_analyzer import describe_upload
from src.model_detector import best_available_model

# ── Singletons ───────────────────────────────────────────────────────────────
ner          = MedicalNER()
expander     = QueryExpander()
drug_checker = DrugChecker()
verifier     = CitationVerifier(threshold=config.CITATION_SIMILARITY_THRESHOLD)

def _make_retriever() -> PubMedRetriever:
    return PubMedRetriever(email=config.PUBMED_EMAIL, api_key=config.NCBI_API_KEY or None)

MODEL_CHOICES = [(info["label"], model_id) for model_id, info in AVAILABLE_MODELS.items()]

# ── Startup model detection (runs once at import, uses env-var / HF Secrets) ─
# This replaces demo.load() — safe because it runs before the UI is built,
# never fires a Gradio event, and can never produce a toast error.
def _startup_detect() -> tuple:
    """
    Detect available models at startup using HF Spaces secrets / env vars.
    Priority: Gemini always beats Groq as the default provider.
    If only a Groq key is present, pick the best confirmed TEXT model, not a vision model.
    """
    gkey = config.GEMINI_API_KEY.strip()
    rkey = config.GROQ_API_KEY.strip()
    if not gkey and not rkey:
        return MODEL_CHOICES, DEFAULT_MODEL
    try:
        best, choices = best_available_model(gkey, rkey)
        if choices:
            # best_available_model returns Gemini models first — so if a Gemini key is
            # set, best is already a Gemini model.  When only Groq key is present, make
            # sure we don't default to a vision model for a text-reasoning task.
            if not gkey:
                # Groq-only: prefer text models; skip vision-specific ones as default
                text_first = next(
                    (mid for _, mid in choices if "scout" not in mid.lower()),
                    best,
                )
                best = text_first
            return choices, best
    except Exception:
        pass
    return MODEL_CHOICES, DEFAULT_MODEL

_INIT_CHOICES, _INIT_MODEL = _startup_detect()

# ── Sample cases ─────────────────────────────────────────────────────────────
SAMPLE_CASES = [
    (
        "Heart Attack",
        "STEMI",
        """57-year-old male with acute crushing chest pain 9/10 for 2 hours, radiating to left arm and jaw.
Associated: diaphoresis, nausea, shortness of breath.
PMH: Hypertension (lisinopril 10 mg), Type 2 DM (metformin 1000 mg BID), Hyperlipidaemia (atorvastatin 40 mg).
Vitals: BP 160/95 mmHg, HR 98 bpm, RR 22/min, SpO2 94% RA, Temp 37.0 C
ECG: ST elevation in II, III, aVF.
Troponin I: 2.8 ng/mL (elevated). D-dimer: 0.3 ug/mL.""",
    ),
    (
        "Night Sweats",
        "Lymphoma?",
        """28-year-old female, 3-week history of fatigue, 4 kg weight loss, night sweats,
painless cervical lymphadenopathy. No fever, cough, or chest pain. No medications.
Uncle had Non-Hodgkin lymphoma.
Vitals: BP 110/70, HR 88, Temp 37.2 C
Labs: WBC 14,500 cells/uL (elevated), Haemoglobin 10.2 g/dL (low),
ESR 68 mm/hr, LDH elevated, CRP 32 mg/L.""",
    ),
    (
        "Abdominal Pain",
        "Appendicitis?",
        """22-year-old male with 12-hour periumbilical pain migrating to right lower quadrant.
Fever 38.4 C, nausea, vomiting, anorexia.
On exam: Rebound tenderness RLQ, positive Rovsing's sign, guarding.
Labs: WBC 16,200 with 88% neutrophils, CRP 85 mg/L.
No significant PMH, no medications.""",
    ),
]

# ── HTML progress renderer ────────────────────────────────────────────────────
_STEP_ICONS = ["", "🔬", "📡", "🧮", "💊", "🧠", "✅"]

def _progress_html(step: int, msg: str, total: int = 6) -> str:
    pct = int((step / total) * 100)
    icon = _STEP_ICONS[step] if step < len(_STEP_ICONS) else "⏳"
    steps_html = ""
    for i in range(1, total + 1):
        if i < step:
            color, bg = "#10b981", "rgba(16,185,129,0.15)"
        elif i == step:
            color, bg = "#2563eb", "rgba(37,99,235,0.15)"
        else:
            color, bg = "#cbd5e1", "rgba(203,213,225,0.3)"
        steps_html += (
            f'<div style="width:28px;height:28px;border-radius:50%;background:{bg};'
            f'border:2px solid {color};display:flex;align-items:center;justify-content:center;'
            f'font-size:0.7rem;font-weight:700;color:{color}">{i}</div>'
        )
    return f"""
<div style="font-family:Inter,-apple-system,sans-serif;background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
  border:1px solid #bae6fd;border-radius:14px;padding:20px 24px;margin:4px 0">
  <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px">
    <div style="width:40px;height:40px;background:linear-gradient(135deg,#1d4ed8,#0891b2);
      border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:1.2rem">{icon}</div>
    <div>
      <div style="font-weight:700;color:#1e3a8a;font-size:1rem">Analysing in progress…</div>
      <div style="color:#0369a1;font-size:0.875rem;margin-top:2px">{msg}</div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:6px;margin-bottom:10px">{steps_html}</div>
  <div style="background:rgba(255,255,255,0.6);border-radius:8px;height:8px;overflow:hidden">
    <div style="background:linear-gradient(90deg,#1d4ed8,#0891b2);height:8px;width:{pct}%;
      border-radius:8px;transition:width 0.4s ease"></div>
  </div>
  <div style="text-align:right;font-size:0.72rem;color:#64748b;margin-top:5px;font-weight:600">
    {pct}% complete</div>
</div>"""

def _error_html(exc: str, tb: str) -> str:
    return f"""
<div style="font-family:Inter,sans-serif;background:linear-gradient(135deg,#fef2f2,#fff5f5);
  border:1.5px solid #fca5a5;border-radius:14px;padding:20px 24px">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
    <span style="font-size:1.4rem">❌</span>
    <span style="font-weight:700;color:#dc2626;font-size:1rem">Error during analysis</span>
  </div>
  <div style="background:#fff;border:1px solid #fecaca;border-radius:8px;
    padding:12px 14px;font-size:0.85rem;color:#7f1d1d;margin-bottom:10px">{exc}</div>
  <details>
    <summary style="cursor:pointer;color:#991b1b;font-size:0.8rem;font-weight:600">
      Show traceback</summary>
    <pre style="background:#1e293b;color:#94a3b8;padding:12px;border-radius:8px;
      font-size:0.75rem;overflow-x:auto;margin-top:8px">{tb}</pre>
  </details>
  <div style="margin-top:10px;font-size:0.82rem;color:#991b1b">
    Check your API key, internet connection, or try a simpler case.</div>
</div>"""

def _missing_key_html(model_id: str, provider: str) -> str:
    if provider == "groq":
        url  = "https://console.groq.com/keys"
        name = "Groq Cloud API Key"
        color = "#f55036"
    else:
        url  = "https://aistudio.google.com"
        name = "Gemini / Gemma API Key"
        color = "#4285f4"
    return f"""
<div style="font-family:Inter,sans-serif;background:linear-gradient(135deg,#fffbeb,#fef3c7);
  border:1.5px solid #fcd34d;border-radius:14px;padding:20px 24px">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px">
    <span style="font-size:1.3rem">🔑</span>
    <span style="font-weight:700;color:#92400e;font-size:1rem">API key required for
      <code style="background:#fde68a;padding:2px 8px;border-radius:6px;
        font-size:0.88rem">{model_id}</code></span>
  </div>
  <div style="font-size:0.88rem;color:#78350f">
    Please paste your <strong style="color:{color}">{name}</strong> in the field above.<br>
    Get a free key at:
    <a href="{url}" target="_blank"
       style="color:{color};font-weight:700;text-decoration:none">{url}</a>
  </div>
</div>"""

# ── Upload handler ────────────────────────────────────────────────────────────
def handle_file_upload(file_path, gemini_key: str, groq_key: str):
    """Analyze an uploaded image or PDF and populate the case text box."""
    if not file_path:
        return gr.update(), "", False
    # Gradio 5 may return a dict with a 'path' key or a plain string
    path = file_path.get("path", file_path) if isinstance(file_path, dict) else file_path
    # Fall back to env-var / HF Secrets when the UI input is empty
    g_key = (gemini_key or "").strip() or config.GEMINI_API_KEY
    r_key = (groq_key or "").strip() or config.GROQ_API_KEY
    desc, status = describe_upload(path, g_key, r_key)
    icon = "✅" if desc else "❌"
    status_html = (
        f'<div style="font-family:Inter,sans-serif;font-size:0.82rem;padding:6px 10px;'
        f'border-radius:8px;margin-top:4px;'
        f'background:{"#f0fdf4;color:#15803d;border:1px solid #86efac" if desc else "#fef2f2;color:#dc2626;border:1px solid #fca5a5"}">'
        f'{icon} {status}</div>'
    )
    return (gr.update(value=desc) if desc else gr.update()), status_html, bool(desc)


def _analyze_if_uploaded(flag, case_text, model_id, gemini_key, groq_key, max_papers):
    """Auto-trigger full analysis after a successful file upload."""
    if not flag:
        return  # No description was generated — nothing to analyse

    # Check that at least one API key is available before proceeding
    provider = _detect_provider(model_id)
    key = ((groq_key or "").strip() if provider == "groq" else (gemini_key or "").strip())
    key = key or (config.GROQ_API_KEY if provider == "groq" else config.GEMINI_API_KEY)
    if not key:
        yield _missing_key_html(model_id, provider)
        return

    yield from analyze_case(case_text, model_id, gemini_key, groq_key, max_papers)


# ── Dynamic model detection ───────────────────────────────────────────────────
# Minimum key length before we bother hitting the live API (avoids errors while typing)
_MIN_KEY_LEN = 20

def update_model_choices(gemini_key: str, groq_key: str):
    """Called when a user types an API key — detects live models and updates the dropdown."""
    try:
        gkey = (gemini_key or "").strip()
        rkey = (groq_key or "").strip()

        if not gkey and not rkey:
            all_static = [(info["label"], mid) for mid, info in AVAILABLE_MODELS.items()]
            return gr.update(choices=all_static, value=DEFAULT_MODEL), ""

        # Skip live API call while still typing (partial key → 401 / timeout errors)
        g_ready = gkey if len(gkey) >= _MIN_KEY_LEN else ""
        r_ready = rkey if len(rkey) >= _MIN_KEY_LEN else ""

        if not g_ready and not r_ready:
            hint = '<div style="font-family:Inter,sans-serif;font-size:0.78rem;padding:5px 10px;border-radius:8px;margin-top:4px;background:#fffbeb;color:#92400e;border:1px solid #fde68a">⏳ Keep typing your API key…</div>'
            return gr.update(), hint

        best, choices = best_available_model(g_ready, r_ready)
        n = len(choices)
        providers = []
        if g_ready:
            providers.append("Gemini")
        if r_ready:
            providers.append("Groq")
        prov_str = " + ".join(providers) if providers else "static list"
        model_label = next((lbl for lbl, mid in choices if mid == best), best)
        status_html = (
            f'<div style="font-family:Inter,sans-serif;font-size:0.78rem;padding:5px 10px;'
            f'border-radius:8px;margin-top:4px;background:#f0fdf4;color:#15803d;'
            f'border:1px solid #86efac">✅ {n} model{"s" if n != 1 else ""} detected from {prov_str}'
            f' — auto-selected: <strong>{model_label}</strong></div>'
        )
        return gr.update(choices=choices, value=best), status_html
    except Exception:
        # Never let a detection failure become a Gradio toast error
        all_static = [(info["label"], mid) for mid, info in AVAILABLE_MODELS.items()]
        return gr.update(choices=all_static, value=DEFAULT_MODEL), ""


# ── Core analysis function (streaming generator) ─────────────────────────────
def analyze_case(
    case_text: str,
    model_id: str,
    gemini_api_key: str,
    groq_api_key: str,
    max_papers: int,
) -> Generator[str, None, None]:

    if not case_text.strip():
        yield "<p style='color:#6b7280;font-style:italic'>Please enter a patient case description.</p>"
        return

    provider = _detect_provider(model_id)
    api_key  = (groq_api_key if provider == "groq" else gemini_api_key).strip()
    api_key  = api_key or (config.GROQ_API_KEY if provider == "groq" else config.GEMINI_API_KEY)

    if not api_key:
        yield _missing_key_html(model_id, provider)
        return

    model_label = AVAILABLE_MODELS.get(model_id, {}).get("label", model_id)

    try:
        yield _progress_html(1, "Extracting medical entities from case text…")
        entities = ner.extract(case_text)

        yield _progress_html(2, "Expanding queries and searching PubMed (~10 s)…")
        retriever = _make_retriever()
        queries   = expander.expand(entities)
        papers    = retriever.multi_query_search(queries, max_per_query=max(5, max_papers // 3))
        if not papers:
            all_terms = ner.to_search_terms(entities)
            if all_terms:
                broad = " OR ".join(f'"{t}"' for t in all_terms[:5])
                papers = retriever.search(broad, max_results=max_papers)

        relevant_papers = papers
        if papers:
            yield _progress_html(3, f"Building embeddings & ranking {len(papers)} papers…")
            engine = EmbeddingEngine.get_instance(config.EMBEDDING_MODEL)
            embs   = engine.encode([f"{p['title']} {p['abstract'][:300]}" for p in papers])
            q_emb  = engine.encode([case_text[:512]])[0]
            store  = VectorStore()
            store.add_documents(papers, embs)
            top    = store.search(q_emb, top_k=min(10, len(papers)))
            relevant_papers = [p for p, s in top if s > 0.2] or papers[:10]

        yield _progress_html(4, "Checking drug–drug interactions via RxNorm API…")
        meds = entities.get("medications", [])
        drug_interactions = (
            drug_checker.check_interactions(meds)
            if len(meds) >= 2
            else {"interactions": [], "drugs_checked": meds}
        )

        yield _progress_html(5, f"Multi-agent reasoning with {model_label} (30–60 s)…")
        pipeline = ReasoningPipeline(api_key=api_key, model=model_id)
        result   = pipeline.generate_differential_diagnosis(
            case_text=case_text,
            entities=entities,
            evidence_papers=relevant_papers,
        )

        yield _progress_html(6, "Verifying citations with cosine similarity…")
        try:
            result["diagnoses"] = verifier.verify_citations(
                result.get("diagnoses", []), relevant_papers
            )
        except Exception:
            pass

        md_output = format_diagnosis_output(
            entities=entities,
            papers=relevant_papers,
            reasoning_result=result,
            drug_interactions=drug_interactions,
        )
        if provider == "gemini":
            prov_badge  = '<span style="background:#fde68a;color:#92400e;padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:700">Google AI — Gemini</span>'
            prov_header = '<span style="background:#4285f4;color:#fff;padding:3px 10px;border-radius:12px;font-size:0.8rem;font-weight:700">🇬 Google AI — Gemini</span>'
        else:
            prov_badge  = '<span style="background:#fce7e3;color:#c2410c;padding:2px 8px;border-radius:10px;font-size:0.75rem;font-weight:700">Groq Cloud — LLM</span>'
            prov_header = '<span style="background:#f55036;color:#fff;padding:3px 10px;border-radius:12px;font-size:0.8rem;font-weight:700">⚡ Groq Cloud</span>'

        model_banner = (
            f'\n\n<div style="font-family:Inter,sans-serif;background:linear-gradient(135deg,#f8fafc,#f1f5f9);'
            f'border:1px solid #e2e8f0;border-radius:10px;padding:10px 16px;margin-bottom:12px;'
            f'display:flex;align-items:center;gap:10px;flex-wrap:wrap">'
            f'{prov_header}'
            f'<code style="background:#f1f5f9;padding:2px 8px;border-radius:6px;font-size:0.82rem;color:#1d4ed8">{model_id}</code>'
            f'<span style="font-size:0.8rem;color:#64748b">·</span>'
            f'<span style="font-size:0.8rem;color:#64748b">{len(relevant_papers)} PubMed papers</span>'
            f'</div>\n\n'
        )
        md_output = model_banner + md_output
        md_output += (
            f"\n\n---\n*Powered by `{model_id}` &nbsp;{prov_badge} &nbsp;·&nbsp; "
            f"{len(relevant_papers)} PubMed papers retrieved*"
        )
        yield md_output

    except Exception as exc:
        yield _error_html(str(exc), traceback.format_exc())


# ── CSS Design System ─────────────────────────────────────────────────────────
_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Base ───────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    background: linear-gradient(160deg, #eef2ff 0%, #f0f9ff 50%, #f0fdf4 100%) !important;
    min-height: 100vh;
}

/* ── Analyse button ─────────────────────────────────── */
#analyze-btn button {
    background: linear-gradient(135deg, #1d4ed8 0%, #0369a1 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.4px !important;
    padding: 15px 20px !important;
    width: 100% !important;
    box-shadow: 0 4px 20px rgba(29,78,216,0.38) !important;
    transition: all 0.2s ease !important;
    position: relative !important;
    overflow: hidden !important;
}
#analyze-btn button::after {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent);
    transform: skewX(-20deg);
    transition: left 0.55s ease;
}
#analyze-btn button:hover::after { left: 140%; }
#analyze-btn button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 30px rgba(29,78,216,0.48) !important;
    background: linear-gradient(135deg, #2563eb 0%, #0891b2 100%) !important;
}
#analyze-btn button:active { transform: translateY(0) !important; }

/* ── Clear button ───────────────────────────────────── */
#clear-btn button {
    background: #fff !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    color: #64748b !important;
    font-weight: 600 !important;
    transition: all 0.18s ease !important;
}
#clear-btn button:hover {
    background: #f8fafc !important;
    border-color: #94a3b8 !important;
    color: #334155 !important;
    transform: translateY(-1px) !important;
}

/* ── Sample case buttons ────────────────────────────── */
#sample-0 button, #sample-1 button, #sample-2 button {
    background: linear-gradient(135deg, #f0f9ff, #dbeafe) !important;
    border: 1.5px solid #93c5fd !important;
    border-radius: 22px !important;
    color: #1d4ed8 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    padding: 6px 16px !important;
    transition: all 0.18s ease !important;
    white-space: nowrap !important;
    letter-spacing: 0.2px !important;
}
#sample-0 button:hover, #sample-1 button:hover, #sample-2 button:hover {
    background: linear-gradient(135deg, #1d4ed8, #0891b2) !important;
    border-color: transparent !important;
    color: #fff !important;
    box-shadow: 0 4px 16px rgba(29,78,216,0.3) !important;
    transform: translateY(-2px) !important;
}

/* ── Patient case textarea ──────────────────────────── */
#case-input textarea {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 12px !important;
    font-size: 0.9rem !important;
    line-height: 1.7 !important;
    color: #0f172a !important;
    padding: 14px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    font-family: 'Inter', sans-serif !important;
    resize: vertical !important;
}
#case-input textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 4px rgba(59,130,246,0.1) !important;
    background: #fff !important;
    outline: none !important;
}
#case-input label, #case-input .label-wrap span {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    color: #374151 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}

/* ── Model dropdown ─────────────────────────────────── */
#model-select label, #model-select .label-wrap span {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    color: #374151 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}

/* ── API key fields ─────────────────────────────────── */
#gemini-key input, #groq-key input {
    background: #f8fafc !important;
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
}
#gemini-key input:focus {
    border-color: #4285f4 !important;
    box-shadow: 0 0 0 3px rgba(66,133,244,0.1) !important;
}
#groq-key input:focus {
    border-color: #f55036 !important;
    box-shadow: 0 0 0 3px rgba(245,80,54,0.1) !important;
}
#gemini-key label, #gemini-key .label-wrap span,
#groq-key label, #groq-key .label-wrap span {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}
#gemini-key .label-wrap span { color: #4285f4 !important; }
#groq-key   .label-wrap span { color: #f55036 !important; }

/* ── Papers slider ──────────────────────────────────── */
#papers-slider input[type=range] { accent-color: #2563eb; cursor: pointer; }
#papers-slider label, #papers-slider .label-wrap span {
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    color: #374151 !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}

/* ── Output markdown ────────────────────────────────── */
#output-panel {
    background: #fff;
    border-radius: 16px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 4px 30px rgba(0,0,0,0.06) !important;
}
#output-panel h1 {
    font-size: 1.5rem !important; font-weight: 800 !important;
    color: #0f172a !important; letter-spacing: -0.3px !important;
}
#output-panel h2 {
    font-size: 1rem !important; font-weight: 700 !important;
    color: #1e3a8a !important;
    border-bottom: 2px solid #f1f5f9 !important;
    padding-bottom: 8px !important; margin-top: 28px !important;
    display: flex; align-items: center; gap: 8px;
}
#output-panel h3 {
    font-size: 0.95rem !important; font-weight: 700 !important;
    color: #1d4ed8 !important; margin-top: 18px !important;
}
#output-panel p  { color: #334155 !important; font-size: 0.9rem !important; line-height: 1.7 !important; }
#output-panel li { color: #334155 !important; font-size: 0.88rem !important; margin: 3px 0 !important; }
#output-panel strong { color: #0f172a !important; font-weight: 700 !important; }
#output-panel a  {
    color: #2563eb !important; text-decoration: none !important;
    font-weight: 600 !important; border-bottom: 1px dotted #93c5fd !important;
}
#output-panel a:hover { border-bottom-style: solid !important; }
#output-panel code {
    background: #f1f5f9 !important; padding: 2px 8px !important;
    border-radius: 6px !important; font-size: 0.82rem !important;
    color: #1d4ed8 !important; font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
}
#output-panel pre code {
    background: #1e293b !important; color: #94a3b8 !important;
    display: block !important; padding: 14px !important;
    border-radius: 10px !important; font-size: 0.8rem !important; overflow-x: auto !important;
}
#output-panel blockquote {
    border-left: 4px solid #ef4444 !important;
    background: linear-gradient(135deg, #fef2f2, #fff5f5) !important;
    padding: 12px 18px !important; border-radius: 0 10px 10px 0 !important;
    margin: 14px 0 !important; color: #991b1b !important; font-weight: 600 !important;
}
#output-panel hr {
    border: none !important; border-top: 2px solid #f1f5f9 !important; margin: 24px 0 !important;
}
#output-panel table {
    width: 100% !important; border-collapse: collapse !important;
    border-radius: 10px !important; overflow: hidden !important; margin: 14px 0 !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
}
#output-panel th {
    background: linear-gradient(135deg, #f1f5f9, #e2e8f0) !important;
    padding: 10px 14px !important; text-align: left !important;
    font-size: 0.76rem !important; font-weight: 700 !important;
    color: #374151 !important; text-transform: uppercase !important; letter-spacing: 0.5px !important;
}
#output-panel td {
    padding: 9px 14px !important; border-bottom: 1px solid #f8fafc !important;
    font-size: 0.87rem !important; color: #334155 !important;
}
#output-panel tr:hover td { background: #f8fafc !important; }

/* ── Accordion ──────────────────────────────────────── */
.gr-accordion { border-radius: 10px !important; border: 1.5px solid #e2e8f0 !important; }

/* ── Responsive ─────────────────────────────────────── */
@media (max-width: 768px) {
    .hdr-title { font-size: 1.4rem !important; }
}
"""

# ── HTML fragments ────────────────────────────────────────────────────────────
_HEADER_HTML = """
<div style="
  background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #0369a1 100%);
  border-radius: 18px; padding: 36px 40px; margin-bottom: 0;
  position: relative; overflow: hidden;
  box-shadow: 0 25px 60px -10px rgba(15,23,42,0.45);
  font-family: Inter, -apple-system, sans-serif;">

  <!-- decorative blobs -->
  <div style="position:absolute;top:-80px;right:-60px;width:320px;height:320px;
    background:radial-gradient(circle,rgba(99,179,237,0.12) 0%,transparent 70%);
    border-radius:50%;pointer-events:none"></div>
  <div style="position:absolute;bottom:-60px;left:-40px;width:260px;height:260px;
    background:radial-gradient(circle,rgba(16,185,129,0.09) 0%,transparent 70%);
    border-radius:50%;pointer-events:none"></div>

  <!-- content -->
  <div style="position:relative;z-index:1">
    <div style="display:flex;align-items:center;gap:18px;margin-bottom:16px">
      <div style="width:60px;height:60px;background:rgba(255,255,255,0.1);
        border:1px solid rgba(255,255,255,0.2);border-radius:16px;
        display:flex;align-items:center;justify-content:center;
        font-size:2rem;backdrop-filter:blur(8px);flex-shrink:0">🏥</div>
      <div>
        <h1 class="hdr-title" style="font-size:2rem;font-weight:800;color:#fff;
          margin:0 0 5px 0;letter-spacing:-0.5px">MedReason-RAG</h1>
        <p style="color:rgba(255,255,255,0.65);margin:0;font-size:0.95rem;font-weight:400">
          Evidence-Grounded Clinical Reasoning &nbsp;·&nbsp; Live PubMed RAG &nbsp;·&nbsp; Multi-Agent AI</p>
      </div>
    </div>

    <!-- badges -->
    <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:24px">
      <span style="background:rgba(16,185,129,0.18);border:1px solid rgba(16,185,129,0.35);
        color:#6ee7b7;padding:4px 13px;border-radius:20px;font-size:0.74rem;font-weight:700;letter-spacing:0.3px">
        📚 PubMed 5M+ Papers</span>
      <span style="background:rgba(99,179,237,0.18);border:1px solid rgba(99,179,237,0.35);
        color:#93c5fd;padding:4px 13px;border-radius:20px;font-size:0.74rem;font-weight:700;letter-spacing:0.3px">
        🤖 8 Free AI Models</span>
      <span style="background:rgba(245,158,11,0.18);border:1px solid rgba(245,158,11,0.35);
        color:#fcd34d;padding:4px 13px;border-radius:20px;font-size:0.74rem;font-weight:700;letter-spacing:0.3px">
        ✨ 100% Free APIs</span>
      <span style="background:rgba(167,139,250,0.18);border:1px solid rgba(167,139,250,0.35);
        color:#c4b5fd;padding:4px 13px;border-radius:20px;font-size:0.74rem;font-weight:700;letter-spacing:0.3px">
        🔬 Real-time RAG</span>
      <span style="background:rgba(239,68,68,0.18);border:1px solid rgba(239,68,68,0.35);
        color:#fca5a5;padding:4px 13px;border-radius:20px;font-size:0.74rem;font-weight:700;letter-spacing:0.3px">
        ⚠️ Educational Use Only</span>
    </div>

    <!-- stats row -->
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">
      <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.14);
        border-radius:12px;padding:14px 16px;text-align:center;backdrop-filter:blur(6px)">
        <div style="font-size:1.5rem;font-weight:800;color:#fff;line-height:1">5M+</div>
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.55);margin-top:4px;font-weight:600">
          Open Papers</div>
      </div>
      <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.14);
        border-radius:12px;padding:14px 16px;text-align:center;backdrop-filter:blur(6px)">
        <div style="font-size:1.5rem;font-weight:800;color:#fff;line-height:1">8</div>
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.55);margin-top:4px;font-weight:600">
          AI Models</div>
      </div>
      <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.14);
        border-radius:12px;padding:14px 16px;text-align:center;backdrop-filter:blur(6px)">
        <div style="font-size:1.5rem;font-weight:800;color:#fff;line-height:1">2</div>
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.55);margin-top:4px;font-weight:600">
          AI Agents</div>
      </div>
      <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.14);
        border-radius:12px;padding:14px 16px;text-align:center;backdrop-filter:blur(6px)">
        <div style="font-size:1.5rem;font-weight:800;color:#fff;line-height:1">$0</div>
        <div style="font-size:0.72rem;color:rgba(255,255,255,0.55);margin-top:4px;font-weight:600">
          API Cost</div>
      </div>
    </div>
  </div>
</div>"""

_SECTION_CASE = """
<div style="font-family:Inter,sans-serif;display:flex;align-items:center;gap:8px;
  font-size:0.72rem;font-weight:700;color:#6b7280;letter-spacing:1.4px;text-transform:uppercase;
  margin-bottom:10px">
  <div style="width:4px;height:18px;background:linear-gradient(to bottom,#2563eb,#0891b2);border-radius:2px"></div>
  PATIENT CASE INPUT
</div>"""

_SECTION_UPLOAD = """
<div style="font-family:Inter,sans-serif;display:flex;align-items:center;gap:8px;
  font-size:0.72rem;font-weight:700;color:#6b7280;letter-spacing:1.4px;text-transform:uppercase;
  margin:14px 0 8px 0">
  <div style="width:4px;height:18px;background:linear-gradient(to bottom,#7c3aed,#c084fc);border-radius:2px"></div>
  UPLOAD IMAGE / PDF &nbsp;<span style="font-weight:400;font-size:0.68rem;color:#94a3b8;letter-spacing:0.5px;
    text-transform:none">(X-ray · MRI · CT · Report — AI will generate description)</span>
</div>"""

_SECTION_SAMPLE = """
<div style="font-family:Inter,sans-serif;display:flex;align-items:center;gap:8px;
  font-size:0.7rem;font-weight:700;color:#94a3b8;letter-spacing:1.2px;text-transform:uppercase;
  margin:14px 0 8px 0">
  <span>⚡</span> QUICK SAMPLE CASES
</div>"""

_SECTION_CONFIG = """
<div style="font-family:Inter,sans-serif;display:flex;align-items:center;gap:8px;
  font-size:0.72rem;font-weight:700;color:#6b7280;letter-spacing:1.4px;text-transform:uppercase;
  margin-bottom:10px">
  <div style="width:4px;height:18px;background:linear-gradient(to bottom,#7c3aed,#a78bfa);border-radius:2px"></div>
  AI CONFIGURATION
</div>"""

_SECTION_OUTPUT = """
<div style="font-family:Inter,sans-serif;display:flex;align-items:center;gap:8px;
  font-size:0.72rem;font-weight:700;color:#6b7280;letter-spacing:1.4px;text-transform:uppercase;
  margin-bottom:4px">
  <div style="width:4px;height:18px;background:linear-gradient(to bottom,#059669,#10b981);border-radius:2px"></div>
  ANALYSIS RESULTS
  <span style="background:#dcfce7;color:#15803d;padding:2px 10px;border-radius:12px;
    font-size:0.68rem;margin-left:4px">LIVE STREAMING</span>
</div>"""

_HOW_IT_WORKS = """
<div style="font-family:Inter,sans-serif;background:linear-gradient(135deg,#f0f9ff,#e0f2fe);
  border:1px solid #bae6fd;border-radius:12px;padding:16px 18px">
  <div style="font-size:0.72rem;font-weight:700;color:#0369a1;letter-spacing:1px;
    text-transform:uppercase;margin-bottom:10px">⚙️ HOW IT WORKS</div>
  <div style="display:flex;flex-direction:column;gap:7px">
    <div style="display:flex;align-items:center;gap:9px;font-size:0.83rem;color:#0c4a6e">
      <div style="width:22px;height:22px;background:linear-gradient(135deg,#1d4ed8,#0891b2);
        border-radius:50%;display:flex;align-items:center;justify-content:center;
        color:#fff;font-size:0.68rem;font-weight:700;flex-shrink:0">1</div>
      Enter detailed patient case</div>
    <div style="display:flex;align-items:center;gap:9px;font-size:0.83rem;color:#0c4a6e">
      <div style="width:22px;height:22px;background:linear-gradient(135deg,#1d4ed8,#0891b2);
        border-radius:50%;display:flex;align-items:center;justify-content:center;
        color:#fff;font-size:0.68rem;font-weight:700;flex-shrink:0">2</div>
      Choose a free AI model</div>
    <div style="display:flex;align-items:center;gap:9px;font-size:0.83rem;color:#0c4a6e">
      <div style="width:22px;height:22px;background:linear-gradient(135deg,#1d4ed8,#0891b2);
        border-radius:50%;display:flex;align-items:center;justify-content:center;
        color:#fff;font-size:0.68rem;font-weight:700;flex-shrink:0">3</div>
      Paste matching API key</div>
    <div style="display:flex;align-items:center;gap:9px;font-size:0.83rem;color:#0c4a6e">
      <div style="width:22px;height:22px;background:linear-gradient(135deg,#1d4ed8,#0891b2);
        border-radius:50%;display:flex;align-items:center;justify-content:center;
        color:#fff;font-size:0.68rem;font-weight:700;flex-shrink:0">4</div>
      Click Analyse — see live results</div>
  </div>
</div>"""

_API_LINKS = """
<div style="font-family:Inter,sans-serif;margin-top:12px">
  <div style="font-size:0.72rem;font-weight:700;color:#6b7280;letter-spacing:1px;
    text-transform:uppercase;margin-bottom:8px">🔑 GET FREE API KEYS</div>
  <a href="https://aistudio.google.com" target="_blank"
     style="display:flex;align-items:center;gap:8px;padding:9px 12px;
       background:linear-gradient(135deg,#f0f9ff,#dbeafe);border:1px solid #93c5fd;
       border-radius:9px;text-decoration:none;margin-bottom:6px;transition:all 0.15s">
    <span style="font-size:1rem">🇬</span>
    <div>
      <div style="font-size:0.8rem;font-weight:700;color:#1d4ed8">Gemini / Gemma</div>
      <div style="font-size:0.7rem;color:#64748b">aistudio.google.com</div>
    </div>
  </a>
  <a href="https://console.groq.com/keys" target="_blank"
     style="display:flex;align-items:center;gap:8px;padding:9px 12px;
       background:linear-gradient(135deg,#fff7ed,#fed7aa);border:1px solid #fdba74;
       border-radius:9px;text-decoration:none;margin-bottom:6px;transition:all 0.15s">
    <span style="font-size:1rem">⚡</span>
    <div>
      <div style="font-size:0.8rem;font-weight:700;color:#c2410c">Groq Cloud</div>
      <div style="font-size:0.7rem;color:#64748b">console.groq.com/keys</div>
    </div>
  </a>
  <a href="https://www.ncbi.nlm.nih.gov/account/" target="_blank"
     style="display:flex;align-items:center;gap:8px;padding:9px 12px;
       background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:1px solid #86efac;
       border-radius:9px;text-decoration:none;transition:all 0.15s">
    <span style="font-size:1rem">🧬</span>
    <div>
      <div style="font-size:0.8rem;font-weight:700;color:#15803d">NCBI PubMed <span style="font-weight:400;color:#64748b">(optional)</span></div>
      <div style="font-size:0.7rem;color:#64748b">Raises rate limit to 10 req/s</div>
    </div>
  </a>
</div>"""

_FOOTER_HTML = """
<div style="font-family:Inter,sans-serif;text-align:center;padding:18px 16px;
  color:#94a3b8;font-size:0.76rem;border-top:1px solid #e2e8f0;margin-top:4px">
  MedReason-RAG &nbsp;·&nbsp; Open-source research prototype &nbsp;·&nbsp;
  Evidence from <a href="https://pubmed.ncbi.nlm.nih.gov" target="_blank"
    style="color:#64748b;font-weight:600;text-decoration:none">PubMed Open Access</a>
  &nbsp;·&nbsp;
  <strong style="color:#dc2626">Not for clinical use</strong>
</div>"""

# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="MedReason-RAG",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ),
    css=_CSS,
) as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML(_HEADER_HTML)

    # ── Main area ─────────────────────────────────────────────────────────────
    with gr.Row(equal_height=False):

        # ── Left column — case input ──────────────────────────────────────────
        with gr.Column(scale=3, min_width=340):
            gr.HTML(_SECTION_CASE)
            case_input = gr.Textbox(
                label="",
                placeholder=(
                    "Enter patient case details:\n"
                    "• Chief complaint & symptom onset\n"
                    "• Vital signs (BP, HR, SpO2, Temp)\n"
                    "• Relevant lab values & ECG findings\n"
                    "• Current medications & past medical history"
                ),
                lines=13,
                value=SAMPLE_CASES[0][2],
                elem_id="case-input",
            )

            # Upload section
            gr.HTML(_SECTION_UPLOAD)
            with gr.Row(equal_height=True):
                upload_file = gr.File(
                    label="",
                    file_types=["image", ".pdf"],
                    file_count="single",
                    type="filepath",
                    scale=3,
                )
            upload_status = gr.HTML(value="")
            did_upload = gr.State(value=False)

            # Sample case buttons
            gr.HTML(_SECTION_SAMPLE)
            with gr.Row():
                sample_btns = []
                for idx, (title, subtitle, _) in enumerate(SAMPLE_CASES):
                    btn = gr.Button(
                        f"{title}  ({subtitle})",
                        size="sm",
                        elem_id=f"sample-{idx}",
                    )
                    sample_btns.append(btn)

        # ── Right column — config + actions ───────────────────────────────────
        with gr.Column(scale=2, min_width=280):
            gr.HTML(_SECTION_CONFIG)

            model_dropdown = gr.Dropdown(
                label="AI MODEL  (auto-updates when you enter an API key)",
                choices=_INIT_CHOICES,
                value=_INIT_MODEL,
                interactive=True,
                elem_id="model-select",
            )
            model_status = gr.HTML(value="")

            with gr.Accordion("🔑  API Keys", open=True):
                gemini_key_input = gr.Textbox(
                    label="GEMINI / GEMMA KEY",
                    placeholder="Paste from aistudio.google.com",
                    type="password",
                    # Pre-fill from HF Spaces secret / env var so the key is available
                    # to analyze_case directly — no separate fallback needed.
                    # Safe: demo.load() is removed so no event fires on page load.
                    value=config.GEMINI_API_KEY,
                    elem_id="gemini-key",
                )
                groq_key_input = gr.Textbox(
                    label="GROQ CLOUD KEY",
                    placeholder="Paste from console.groq.com/keys",
                    type="password",
                    value=config.GROQ_API_KEY,
                    elem_id="groq-key",
                )

            max_papers_slider = gr.Slider(
                label="MAX PUBMED PAPERS",
                minimum=5, maximum=30, value=15, step=5,
                elem_id="papers-slider",
            )

            with gr.Row():
                analyze_btn = gr.Button(
                    "🔬  Analyse Case",
                    variant="primary",
                    scale=3,
                    elem_id="analyze-btn",
                )
                clear_btn = gr.Button("✕  Clear", scale=1, elem_id="clear-btn")

            gr.HTML(_HOW_IT_WORKS)
            gr.HTML(_API_LINKS)

    # ── Output panel ──────────────────────────────────────────────────────────
    gr.HTML(_SECTION_OUTPUT)
    output_md = gr.Markdown(
        value="""
<div style="font-family:Inter,sans-serif;text-align:center;padding:48px 20px;color:#94a3b8">
  <div style="font-size:3rem;margin-bottom:12px">🏥</div>
  <div style="font-size:1rem;font-weight:600;color:#64748b;margin-bottom:6px">
    Ready to analyse</div>
  <div style="font-size:0.85rem">
    Enter a patient case, choose a model, add your API key, then click <strong>Analyse Case</strong>.</div>
</div>""",
        elem_id="output-panel",
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML(_FOOTER_HTML)

    # ── Event wiring ──────────────────────────────────────────────────────────
    analyze_btn.click(
        fn=analyze_case,
        inputs=[case_input, model_dropdown, gemini_key_input, groq_key_input, max_papers_slider],
        outputs=output_md,
    )
    clear_btn.click(
        fn=lambda: ("", """<div style="text-align:center;padding:48px;color:#94a3b8;font-family:Inter,sans-serif">
          <div style="font-size:2rem;margin-bottom:8px">🗑️</div>
          <div style="font-size:0.9rem">Cleared. Enter a new case to begin.</div></div>"""),
        outputs=[case_input, output_md],
    )
    for idx, (btn, (_, _, text)) in enumerate(zip(sample_btns, SAMPLE_CASES)):
        btn.click(fn=lambda t=text: t, outputs=case_input)

    # Upload: analyze file → populate case text box → auto-run full analysis
    upload_file.change(
        fn=handle_file_upload,
        inputs=[upload_file, gemini_key_input, groq_key_input],
        outputs=[case_input, upload_status, did_upload],
    ).then(
        fn=_analyze_if_uploaded,
        inputs=[did_upload, case_input, model_dropdown, gemini_key_input, groq_key_input, max_papers_slider],
        outputs=output_md,
    )

    # API key change: detect live models → update dropdown
    gemini_key_input.change(
        fn=update_model_choices,
        inputs=[gemini_key_input, groq_key_input],
        outputs=[model_dropdown, model_status],
    )
    groq_key_input.change(
        fn=update_model_choices,
        inputs=[gemini_key_input, groq_key_input],
        outputs=[model_dropdown, model_status],
    )

    # Note: model detection for HF Spaces env-var secrets is done at startup
    # (_startup_detect above) — no demo.load() needed, avoids Gradio toast errors.


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
    )
