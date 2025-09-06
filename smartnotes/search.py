
from __future__ import annotations
import numpy as np
from typing import Optional
from .embeddings import encode_texts

def to_uniform_notes(notes_raw: list) -> list[dict]:
    uniform = []
    for n in notes_raw:
        if isinstance(n, str):
            uniform.append({'content': n, 'category': 'General', 'tags': []})
        elif isinstance(n, dict):
            uniform.append({
                'content': n.get('content', ''),
                'category': n.get('category', 'General'),
                'tags': n.get('tags', []) or []
            })
        else:
            raise TypeError(f'Unsupported note type: {type(n)}')
    return uniform

def compute_embeddings(notes: list[dict]) -> np.ndarray:
    texts = [n['content'] for n in notes]
    return encode_texts(texts, device='cpu', normalize=True)

def cosine_search(query: str, notes: list[dict], note_embs: np.ndarray, top_k: int = 5,
                  min_similarity: float = 0.0,
                  category: Optional[str] = None,
                  required_tags: Optional[list[str]] = None) -> list[dict]:
    if not notes or note_embs is None or len(note_embs) == 0:
        return []
    q_emb = encode_texts([query], device='cpu', normalize=True)[0]
    sims = note_embs @ q_emb
    idx_sorted = np.argsort(sims)[::-1]

    results = []
    for idx in idx_sorted:
        sim = float(sims[idx])
        if sim < min_similarity:
            continue
        note = notes[idx]
        if category and note.get('category') != category:
            continue
        if required_tags:
            tags = set(t.lower() for t in (note.get('tags') or []))
            if not set(t.lower() for t in required_tags).issubset(tags):
                continue
        results.append({'index': int(idx), 'note': note, 'similarity': sim})
        if len(results) >= top_k:
            break
    return results
