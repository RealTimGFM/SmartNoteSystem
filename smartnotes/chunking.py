
from __future__ import annotations
from typing import List

def chunk_note(note: str, max_words: int = 50) -> List[str]:
    words = note.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
