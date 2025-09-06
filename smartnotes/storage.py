
from __future__ import annotations
import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "notes.json"

def load_notes(path = DEFAULT_PATH) -> list:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))

def save_notes(notes: list, path = DEFAULT_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
