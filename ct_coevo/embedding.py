"""
embedding.py - Semantic embedding for memory retrieval and tool selection.

Queries are matched against memory titles and tool descriptions by cosine
similarity, and entries above a threshold are returned. The same similarity
function backs tool matching and the memory merging policy.

Harrier-OSS (0.6B) is microsoft/harrier-oss-v1-0.6b and runs locally through
SentenceTransformers. The model is loaded lazily on first use; set the
environment variable HARRIER_MODEL_DIR to point at a local model directory
when a local copy is preferred over downloading the model id.
"""

import os
from typing import List, Optional

import numpy as np


DEFAULT_HARRIER_MODEL = "microsoft/harrier-oss-v1-0.6b"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between row vectors of a and b (both L2-normalized)."""
    return a @ b.T


class HarrierEmbedder:
    """Sentence-embedding wrapper around Harrier-OSS (0.6B).

    Produces L2-normalized embeddings so that cosine similarity reduces to a
    matrix product. The model is instantiated lazily on first use so that the
    framework can be imported without loading the model.
    """

    def __init__(self, model: Optional[str] = None):
        self._model_name = model or os.environ.get("HARRIER_MODEL_DIR") or DEFAULT_HARRIER_MODEL
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "Memory retrieval and tool selection require the Harrier-OSS (0.6B) "
                "embedding model. Install sentence-transformers "
                "(`pip install sentence-transformers`) and, if needed, point "
                "HARRIER_MODEL_DIR at a local copy of the model."
            ) from e
        self._model = SentenceTransformer(self._model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode a list of texts into L2-normalized embeddings (n x d)."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._load()
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def similarity_matrix(self, query_texts: List[str], key_texts: List[str]) -> np.ndarray:
        """Cosine similarity matrix between two text lists."""
        if not query_texts or not key_texts:
            return np.zeros((len(query_texts), len(key_texts)), dtype=np.float32)
        return cosine_similarity(self.encode(query_texts), self.encode(key_texts))
