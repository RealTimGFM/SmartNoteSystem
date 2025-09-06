# Smart Notes — Semantic Search with SBERT & Streamlit

Search your notes by **meaning** (not just keywords). This app converts notes to Sentence‑BERT embeddings and ranks results with cosine similarity. It ships with a clean Streamlit UI and a tiny CLI.

## 🌟 Features
- **Reactive UI** (Streamlit): type to search, results update instantly  
- **Add / import / export** notes (JSON)  
- **Categories & tags** with filters  
- **Chunk** long notes by word count  
- **Save / load** embeddings for faster startup  
- **CLI** for terminal searches  
- CPU‑friendly; GPU optional

---

## 🚀 Quickstart
# Clone
git clone https://github.com/RealTimGFM/SmartNoteSystem.git
cd SmartNoteSystem

# (Recommended) create a virtual env
python -m venv .venv

# Windows
.venv\Scripts\activate

# Install deps
pip install -r requirements.txt

# Run the app
streamlit run app.py
# Open: http://localhost:8501

> **Note:** First run downloads the SBERT model (~100 MB). CPU is fine; ignore CUDA/GPU warnings.

---

## 🧭 Project Structure
```
SmartNoteSystem/
├─ app.py                     # Streamlit UI
├─ cli.py                     # Terminal search tool
├─ smartnotes/
│  ├─ __init__.py
│  ├─ embeddings.py           # SBERT loading + encoding
│  ├─ search.py               # cosine search, filters
│  ├─ storage.py              # load/save notes JSON
│  └─ chunking.py             # split long notes
├─ data/
│  └─ notes.json              # sample notes (edit/replace)
├─ models/                    # optional saved embeddings
├─ requirements.txt
└─ README.md
```
---

## ✍️ Using the App

1. Add notes in the **sidebar** (or import a JSON).  
2. Enter a **query** (e.g., “How do Python lists work?”).  
3. Adjust **Top K** / **Min similarity**; filter by **Category** / **Tags**.  
4. Click **Save notes** to persist to `data/notes.json`.  
5. Optionally **Save embeddings** to `models/note_embeddings.npy` (and **Load** later).


## 🖥️ CLI Usage
python cli.py --query "object oriented programming" --top_k 5 --min_similarity 0.4

---

## ⚙️ How It Works

- **Model:** `all-MiniLM-L6-v2` (Sentence-Transformers) → 384‑dim embeddings  
- **Similarity:** cosine similarity (dot product on L2‑normalized vectors)  
- **Ranking:** sort by similarity, highest first

---

## 🧩 Tips & Performance

- Keep notes **focused** on a single concept for best matches.  
- **Chunk** long documents to improve recall.  
- Save embeddings to `models/` to avoid recomputation after restart.  
- Use **Category** and **Tags** for quick metadata filtering.

---

## ❓ Troubleshooting

- **Model download fails** → check internet; retry; `streamlit cache clear`  
- **No results** → lower Min similarity, add more notes, or chunk long text  
- **High RAM** → keep notes concise; save & load embeddings  
- **Windows symlink warnings** → safe to ignore

---

## 📜 License
MIT — do anything
