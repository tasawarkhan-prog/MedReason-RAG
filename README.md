<div align="center">

# 🏥 MedReason-RAG

### Evidence-Grounded Clinical Reasoning · Live PubMed Search · Multi-Agent AI

[![Live Demo](https://img.shields.io/badge/🤗%20HuggingFace-Live%20Demo-yellow?style=for-the-badge)](https://huggingface.co/spaces/Tasawar-prog1/MedReason-RAG)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-6.x-orange?style=for-the-badge)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![API Cost](https://img.shields.io/badge/API%20Cost-%240-brightgreen?style=for-the-badge)](https://aistudio.google.com)

**A free, open-source medical AI assistant that searches 5M+ PubMed papers in real-time,
runs two competing AI agents (Diagnostician vs Devil's Advocate),
and generates evidence-grounded differential diagnoses — all at zero API cost.**

[🚀 Try Live Demo](https://huggingface.co/spaces/Tasawar-prog1/MedReason-RAG) · [📋 Report Bug](../../issues) · [💡 Request Feature](../../issues)

</div>

---

## ✨ What Makes This Different

| Feature | MedReason-RAG | Typical Medical AI |
|---------|:---:|:---:|
| Live PubMed search (real papers) | ✅ | ❌ Static knowledge |
| Two competing AI agents | ✅ | ❌ Single model |
| Drug interaction checker | ✅ | ❌ |
| Upload X-ray / MRI / PDF | ✅ | ❌ |
| 100% Free APIs | ✅ | ❌ Paid |
| Citation verification | ✅ | ❌ |

---

## 🎯 Key Features

### 🔬 Multi-Agent Reasoning
Two AI agents debate each diagnosis:
- **Diagnostician** — generates 5 differential diagnoses with confidence scores
- **Devil's Advocate** — challenges each diagnosis and flags missed critical conditions

### 📡 Live PubMed RAG
- Searches PubMed in real-time for relevant papers
- Embeds and ranks papers using semantic similarity
- Grounds every diagnosis with real PMID citations

### 🖼️ Medical Image & PDF Analysis
- Upload **X-ray, MRI, CT scan, ECG, lab reports, pathology slides**
- AI writes a full clinical description from the image
- Automatically continues to full diagnosis pipeline

### 💊 Drug Interaction Checker
- Detects dangerous drug–drug interactions from patient medications
- Powered by the free RxNorm API (no key required)

### 🤖 8 Free AI Models
Automatically detects what your API key can access:

| Provider | Models |
|----------|--------|
| Google Gemini | Gemini 2.5 Flash, 2.0 Flash, 1.5 Flash, Gemma 3 27B |
| Groq Cloud | LLaMA 3.3 70B, LLaMA 4 Scout 17B (Vision), Mixtral 8x7B |

---

## 🏗️ Architecture
## 🏗️ Architecture

```mermaid
flowchart TD
    A[📝 Patient Case Text] --> C
    B[🖼️ Upload Image / PDF] --> B1[AI Vision Analysis\nGemini / LLaMA 4 Scout]
    B1 --> B2[Clinical Description]
    B2 --> C

    C[🔬 Medical NER\nEntity Extraction] --> D
    D[🔍 Query Expansion] --> E
    E[📡 Live PubMed Search\n5M+ Papers] --> F
    F[🧮 Semantic Embedding\n+ Vector Ranking] --> G

    subgraph Agents [🧠 Multi-Agent Reasoning]
        G --> H[Agent 1 — Diagnostician\n5 Differential Diagnoses]
        G --> I[Agent 2 — Devil's Advocate\nChallenges Each Diagnosis]
        H --> J[Final Debate & Consensus]
        I --> J
    end

    J --> K[✅ Citation Verification\nCosine Similarity]
    J --> L[💊 Drug Interaction Check\nRxNorm API]

    K --> M[📋 Evidence-Grounded\nClinical Report with PMIDs]
    L --> M

    style Agents fill:#1e3a5f,color:#fff,stroke:#4a9eff
    style M fill:#1a4731,color:#fff,stroke:#4ade80
    style B1 fill:#4a1942,color:#fff,stroke:#c084fc
