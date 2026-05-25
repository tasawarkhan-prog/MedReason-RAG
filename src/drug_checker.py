from __future__ import annotations
import requests
from typing import List, Dict, Optional


class DrugChecker:
    """
    Checks drug-drug interactions using the free RxNorm API (no key required).
    Falls back gracefully if the API is unavailable.
    """

    _RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"

    def _get_rxcui(self, drug_name: str) -> Optional[str]:
        """Resolve a drug name to its RxNorm concept ID."""
        try:
            url = f"{self._RXNORM_BASE}/rxcui.json"
            resp = requests.get(url, params={"name": drug_name}, timeout=6)
            ids = resp.json().get("idGroup", {}).get("rxnormId", [])
            return ids[0] if ids else None
        except Exception:
            return None

    def check_interactions(self, drug_names: List[str]) -> Dict:
        """Return interaction summary for the supplied list of drug names."""
        if len(drug_names) < 2:
            return {"interactions": [], "drugs_checked": drug_names}

        rxcuis = []
        for drug in drug_names[:6]:
            rxcui = self._get_rxcui(drug)
            if rxcui:
                rxcuis.append(rxcui)

        if len(rxcuis) < 2:
            return {
                "interactions": [],
                "drugs_checked": drug_names,
                "note": "Could not resolve drug names to RxCUI identifiers.",
            }

        try:
            url = f"{self._RXNORM_BASE}/interaction/list.json"
            resp = requests.get(url, params={"rxcuis": " ".join(rxcuis)}, timeout=10)
            data = resp.json()

            interactions = []
            for group in data.get("fullInteractionTypeGroup", []):
                for itype in group.get("fullInteractionType", []):
                    for pair in itype.get("interactionPair", []):
                        concepts = pair.get("interactionConcept", [])
                        drugs = (
                            [
                                concepts[0]["minConceptItem"]["name"],
                                concepts[1]["minConceptItem"]["name"],
                            ]
                            if len(concepts) >= 2
                            else []
                        )
                        interactions.append({
                            "description": pair.get("description", ""),
                            "severity": pair.get("severity", "Unknown"),
                            "drugs": drugs,
                        })

            return {
                "interactions": interactions[:10],
                "drugs_checked": drug_names,
            }

        except Exception as e:
            return {
                "interactions": [],
                "drugs_checked": drug_names,
                "error": str(e),
            }
