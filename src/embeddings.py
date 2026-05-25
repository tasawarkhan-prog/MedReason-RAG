from __future__ import annotations
from typing import List, Optional
import numpy as np


class EmbeddingEngine:
    """Thin wrapper around SentenceTransformer with a process-level singleton."""

    _instance: Optional[EmbeddingEngine] = None

    @classmethod
    def get_instance(cls, model_name: str = "all-MiniLM-L6-v2") -> EmbeddingEngine:
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Deferred import so the class can be imported without heavy deps
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Return L2-normalised embeddings, shape (N, dim)."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
