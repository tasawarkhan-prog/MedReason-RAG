from __future__ import annotations
from typing import List, Dict
import numpy as np


class CitationVerifier:
    """
    Verifies whether retrieved PubMed evidence actually supports a diagnostic claim
    using cosine similarity of sentence embeddings (no separate NLI model needed).
    """

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def verify(self, claim: str, evidence: str) -> Dict:
        """Return support verdict + similarity score for a single claim/evidence pair."""
        from src.embeddings import EmbeddingEngine
        engine = EmbeddingEngine.get_instance()
        embeddings = engine.encode([claim[:512], evidence[:512]])
        similarity = float(np.dot(embeddings[0], embeddings[1]))
        supported = similarity >= self.threshold
        return {
            "supported": supported,
            "confidence": round(similarity, 3),
            "label": "entailment" if supported else "neutral",
        }

    def verify_citations(self, diagnoses: List[Dict], papers: List[Dict]) -> List[Dict]:
        """Attach verification results to each citation in every diagnosis."""
        paper_by_pmid = {p["pmid"]: p for p in papers}
        verified_diagnoses = []
        for dx in diagnoses:
            claim = dx.get("reasoning", "")
            verified_citations = []
            for cit in dx.get("citations", [])[:3]:
                pmid = cit.get("pmid", "")
                paper = paper_by_pmid.get(pmid)
                if paper and claim:
                    try:
                        result = self.verify(claim, paper["abstract"])
                    except Exception:
                        result = {"supported": False, "confidence": 0.0, "label": "error"}
                    verified_citations.append({**cit, "verification": result})
                else:
                    verified_citations.append(cit)
            verified_diagnoses.append({**dx, "citations": verified_citations})
        return verified_diagnoses
