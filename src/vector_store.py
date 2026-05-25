from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np


class VectorStore:
    """In-memory cosine similarity search using numpy (no external vector DB needed)."""

    def __init__(self):
        self._embeddings: np.ndarray | None = None
        self._documents: List[Dict] = []

    def add_documents(self, documents: List[Dict], embeddings: np.ndarray):
        self._documents = list(documents)
        # Ensure float32 and L2-normalised for dot-product = cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self._embeddings = (embeddings / norms).astype(np.float32)

    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Dict, float]]:
        if self._embeddings is None or len(self._documents) == 0:
            return []

        q = query_embedding.astype(np.float32)
        q = q / max(np.linalg.norm(q), 1e-9)

        scores = self._embeddings @ q  # (N,)
        k = min(top_k, len(self._documents))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self._documents[i], float(scores[i])) for i in top_indices]

    def reset(self):
        self._embeddings = None
        self._documents = []
