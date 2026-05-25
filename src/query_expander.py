from typing import Dict, List

# Curated medical knowledge expansion — symptom → likely differential conditions
_EXPANSIONS: Dict[str, List[str]] = {
    "chest pain": ["angina pectoris", "myocardial infarction", "aortic dissection", "pulmonary embolism", "GERD"],
    "shortness of breath": ["heart failure", "pneumonia", "COPD exacerbation", "pulmonary embolism", "asthma"],
    "dyspnea": ["heart failure", "pneumonia", "pulmonary embolism", "COPD", "anemia"],
    "fever": ["bacterial infection", "viral infection", "sepsis", "malignancy", "autoimmune disease"],
    "headache": ["migraine", "subarachnoid hemorrhage", "meningitis", "hypertensive urgency", "tension headache"],
    "dizziness": ["BPPV", "vestibular neuritis", "orthostatic hypotension", "cardiac arrhythmia", "anemia"],
    "abdominal pain": ["appendicitis", "cholecystitis", "pancreatitis", "bowel obstruction", "peptic ulcer disease"],
    "edema": ["heart failure", "nephrotic syndrome", "hepatic cirrhosis", "deep vein thrombosis", "hypoalbuminemia"],
    "syncope": ["vasovagal syncope", "cardiac arrhythmia", "orthostatic hypotension", "pulmonary embolism", "AVNRT"],
    "weight loss": ["malignancy", "hyperthyroidism", "diabetes mellitus", "inflammatory bowel disease", "tuberculosis"],
    "night sweats": ["lymphoma", "tuberculosis", "HIV", "menopause", "brucellosis"],
    "fatigue": ["anemia", "hypothyroidism", "heart failure", "depression", "chronic fatigue syndrome"],
    "palpitations": ["atrial fibrillation", "SVT", "ventricular tachycardia", "hyperthyroidism", "anxiety"],
    "hemoptysis": ["tuberculosis", "lung cancer", "pulmonary embolism", "bronchiectasis", "pneumonia"],
    "hematuria": ["urinary tract infection", "renal cell carcinoma", "bladder cancer", "nephrolithiasis", "glomerulonephritis"],
    "jaundice": ["hepatitis", "biliary obstruction", "hemolytic anemia", "cirrhosis", "pancreatic cancer"],
}


class QueryExpander:
    """Expands medical entities into PubMed-ready queries using clinical knowledge."""

    def expand(self, entities: Dict[str, List[str]]) -> List[str]:
        queries: List[str] = []

        symptoms = entities.get("symptoms", [])
        labs = entities.get("labs", [])
        diagnoses = entities.get("diagnoses", [])

        # Query 1: Primary symptom-based differential
        if symptoms:
            symptom_terms = " OR ".join(f'"{s}"[Title/Abstract]' for s in symptoms[:3])
            queries.append(f"({symptom_terms}) AND (differential diagnosis OR clinical features)")

        # Query 2: Lab-based evidence
        if labs:
            lab_terms = " OR ".join(f'"{l}"[Title/Abstract]' for l in labs[:3])
            queries.append(f"({lab_terms}) AND (diagnosis OR sensitivity specificity)")

        # Query 3: Expanded conditions from symptom knowledge
        expanded: List[str] = []
        for symptom in symptoms:
            expanded.extend(_EXPANSIONS.get(symptom.lower(), []))
        if diagnoses:
            expanded.extend(diagnoses[:3])

        if expanded:
            unique_expanded = list(dict.fromkeys(expanded))[:6]
            cond_terms = " OR ".join(f'"{c}"' for c in unique_expanded)
            queries.append(f"({cond_terms}) AND (systematic review OR meta-analysis OR clinical trial)")

        # Fallback: broad search from all entities
        if not queries:
            all_terms = symptoms + labs + diagnoses
            if all_terms:
                fallback = " OR ".join(f'"{t}"' for t in all_terms[:5])
                queries.append(f"({fallback})")

        return queries[:3]
