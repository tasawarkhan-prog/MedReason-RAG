import re
from typing import Dict, List

# Regex pattern banks for medical entity extraction
_SYMPTOMS = (
    r"chest pain|shortness of breath|dyspnea|palpitations|syncope|fever|chills|"
    r"nausea|vomiting|diarrhea|constipation|abdominal pain|headache|dizziness|"
    r"fatigue|weakness|weight loss|night sweats|cough|hemoptysis|edema|"
    r"diaphoresis|tachycardia|bradycardia|hypotension|hypertension|"
    r"confusion|altered mental status|lethargy|malaise|anorexia|dysphagia|"
    r"hematuria|dysuria|jaundice|pruritus|rash|arthralgia|myalgia"
)

_LABS = (
    r"troponin|BNP|pro-?BNP|creatinine|BUN|glucose|HbA1c|TSH|T3|T4|"
    r"hemoglobin|hematocrit|WBC|platelet|sodium|potassium|chloride|"
    r"bicarbonate|calcium|magnesium|phosphorus|ALT|AST|bilirubin|"
    r"albumin|INR|PT|PTT|D-?dimer|CRP|ESR|procalcitonin|lactate|"
    r"lipase|amylase|uric acid|ferritin|iron|TIBC|B12|folate"
)

_DIAGNOSES = (
    r"diabetes|hypertension|heart failure|COPD|asthma|pneumonia|"
    r"myocardial infarction|\bMI\b|stroke|DVT|pulmonary embolism|\bPE\b|"
    r"sepsis|appendicitis|cholecystitis|pancreatitis|cirrhosis|"
    r"renal failure|\bAKI\b|\bCKD\b|atrial fibrillation|AF|AFIB|"
    r"hypothyroidism|hyperthyroidism|anemia|leukemia|lymphoma|"
    r"carcinoma|cancer|aortic dissection|meningitis|encephalitis"
)

_MEDICATIONS = (
    r"aspirin|metformin|lisinopril|atorvastatin|metoprolol|amlodipine|"
    r"omeprazole|levothyroxine|albuterol|warfarin|heparin|insulin|"
    r"amoxicillin|azithromycin|ceftriaxone|vancomycin|prednisone|"
    r"furosemide|spironolactone|digoxin|amiodarone|rivaroxaban|apixaban|"
    r"losartan|valsartan|hydrochlorothiazide|amlodipine|rosuvastatin|"
    r"clopidogrel|ticagrelor|enoxaparin|morphine|fentanyl|acetaminophen|"
    r"ibuprofen|naproxen|pantoprazole|esomeprazole|sertraline|fluoxetine"
)

_VITALS = (
    r"BP\s*[\d/]+|HR\s*[\d]+|RR\s*[\d]+|"
    r"O2\s*sat\s*[\d]+%?|SpO2\s*[\d]+%?|"
    r"Temp\s*[\d.]+|temperature\s*[\d.]+"
)


class MedicalNER:
    """Regex-based medical named entity recogniser — no external model required."""

    def __init__(self):
        flags = re.IGNORECASE
        self._patterns: Dict[str, re.Pattern] = {
            "symptoms": re.compile(r"\b(?:" + _SYMPTOMS + r")\b", flags),
            "labs": re.compile(r"\b(?:" + _LABS + r")\b", flags),
            "diagnoses": re.compile(r"\b(?:" + _DIAGNOSES + r")\b", flags),
            "medications": re.compile(r"\b(?:" + _MEDICATIONS + r")\b", flags),
        }

    def extract(self, text: str) -> Dict[str, List[str]]:
        """Return deduplicated entity lists per category."""
        result: Dict[str, List[str]] = {}
        for entity_type, pattern in self._patterns.items():
            matches = {m.group(0).lower() for m in pattern.finditer(text)}
            result[entity_type] = sorted(matches)
        return result

    def to_search_terms(self, entities: Dict[str, List[str]]) -> List[str]:
        """Flatten all entities to a single list for query building."""
        terms = []
        for lst in entities.values():
            terms.extend(lst)
        return terms
