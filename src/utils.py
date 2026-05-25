from typing import Dict, List

_LIKELIHOOD_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}
_SEVERITY_EMOJI = {"major": "🔴", "moderate": "🟡", "minor": "🟢"}


def _confidence_bar(score: float) -> str:
    filled = int(min(max(score, 0.0), 1.0) * 10)
    return "█" * filled + "░" * (10 - filled) + f"  {score:.0%}"


def format_diagnosis_output(
    entities: Dict,
    papers: List[Dict],
    reasoning_result: Dict,
    drug_interactions: Dict,
) -> str:
    out: List[str] = []

    # ── Header ──────────────────────────────────────────────────────────
    out.append("# 🏥 MedReason-RAG — Evidence-Grounded Clinical Reasoning")
    out.append(
        "> ⚠️ **EDUCATIONAL USE ONLY — NOT FOR CLINICAL DECISIONS.** "
        "Always consult a qualified healthcare professional.\n"
    )

    # ── Extracted entities ───────────────────────────────────────────────
    out.append("## 🔍 Extracted Medical Entities")
    for label, key in [
        ("Symptoms", "symptoms"),
        ("Lab Values", "labs"),
        ("Medications", "medications"),
        ("Known Diagnoses", "diagnoses"),
    ]:
        items = entities.get(key, [])
        value = ", ".join(items) if items else "*none detected*"
        out.append(f"- **{label}:** {value}")
    out.append("")

    # ── Retrieved evidence ───────────────────────────────────────────────
    out.append(f"## 📚 Retrieved PubMed Evidence  ({len(papers)} papers)")
    for i, p in enumerate(papers[:6]):
        title = p.get("title", "")[:90]
        year = p.get("year", "")
        pmid = p.get("pmid", "")
        url = p.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        out.append(f"{i+1}. **{title}...** ({year}) — [PMID {pmid}]({url})")
    out.append("")

    # ── Critical alert ───────────────────────────────────────────────────
    critical = (reasoning_result.get("devils_advocate") or {}).get("critical_alert", "")
    if critical:
        out.append(f"## 🚨 Critical Alert\n> {critical}\n")

    # ── Differential diagnoses ───────────────────────────────────────────
    out.append("## 📊 Differential Diagnoses")
    diagnoses = reasoning_result.get("diagnoses", [])
    if not diagnoses:
        out.append("*No diagnoses generated — please check your API key and try again.*")
    for i, dx in enumerate(diagnoses):
        likelihood = dx.get("likelihood", "Medium")
        emoji = _LIKELIHOOD_EMOJI.get(likelihood.lower(), "⚪")
        score = float(dx.get("confidence_score", 0.5))

        out.append(
            f"\n### {i+1}. {emoji} {dx.get('condition', 'Unknown')}  "
            f"*(Likelihood: {likelihood})*"
        )
        out.append(f"**Confidence:** `{_confidence_bar(score)}`")

        features = dx.get("supporting_features", [])
        if features:
            out.append(f"**Supporting Features:** {', '.join(features[:4])}")

        reasoning = dx.get("reasoning", "")
        if reasoning:
            out.append(f"**Reasoning:** {reasoning[:350]}")

        citations = dx.get("citations", [])
        if citations:
            out.append("**Evidence Citations:**")
            for cit in citations[:3]:
                pmid = cit.get("pmid", "")
                relevance = cit.get("relevance", "")[:100]
                v = cit.get("verification", {})
                icon = "✅" if v.get("supported") else "⚠️"
                conf = v.get("confidence", "")
                conf_str = f" *(sim={conf:.2f})*" if isinstance(conf, float) else ""
                out.append(
                    f"  - {icon} [PMID {pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/) "
                    f"— {relevance}{conf_str}"
                )

        tests = dx.get("confirmatory_tests", [])
        if tests:
            out.append(f"**Recommended Tests:** {', '.join(tests[:4])}")

    # ── Devil's Advocate ─────────────────────────────────────────────────
    out.append("\n## 🎭 Devil's Advocate Analysis")
    challenges = (reasoning_result.get("devils_advocate") or {}).get("challenges", [])
    if not challenges:
        out.append("*Not available.*")
    for ch in challenges[:4]:
        cond = ch.get("condition", "")
        if not cond:
            continue
        out.append(f"\n**{cond}**")
        if ch.get("contradicting_evidence"):
            out.append(f"- ⚠️ Counter-evidence: {ch['contradicting_evidence'][:200]}")
        if ch.get("missed_critical"):
            out.append(f"- 🚨 Don't miss: {ch['missed_critical'][:200]}")

    # ── Drug interactions ────────────────────────────────────────────────
    interactions = drug_interactions.get("interactions", [])
    if interactions:
        out.append("\n## 💊 Drug Interaction Alerts")
        for inter in interactions[:5]:
            sev = inter.get("severity", "Unknown")
            icon = _SEVERITY_EMOJI.get(sev.lower(), "⚪")
            drugs = " ↔ ".join(inter.get("drugs", []))
            desc = inter.get("description", "")[:180]
            out.append(f"- {icon} **{sev}**  |  {drugs}  |  {desc}")

    # ── Footer ───────────────────────────────────────────────────────────
    out.append("\n---")
    out.append(
        "*MedReason-RAG is an open-source research prototype. "
        "Evidence is retrieved live from PubMed Open Access. "
        "Outputs are for educational demonstration only.*"
    )

    return "\n".join(out)
