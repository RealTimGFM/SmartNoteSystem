
from __future__ import annotations
from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL = None

def get_model(device: str = 'cpu') -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
        try:
            _MODEL.to(device)
        except Exception:
            pass
    return _MODEL

def l2_normalize(X: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=axis, keepdims=True) + eps
    return X / norms

def encode_texts(texts: list[str], device: str = 'cpu', normalize: bool = True) -> np.ndarray:
    model = get_model(device=device)
    embs = model.encode(texts, show_progress_bar=False)
    if normalize:
        if embs.ndim == 1:
            embs = embs / (np.linalg.norm(embs) + 1e-12)
        else:
            embs = l2_normalize(embs, axis=1)
    return embs
