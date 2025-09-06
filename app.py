
import json
import hashlib
from pathlib import Path
import numpy as np
import streamlit as st

from smartnotes.storage import load_notes, save_notes
from smartnotes.search import to_uniform_notes, compute_embeddings, cosine_search
from smartnotes.chunking import chunk_note

st.set_page_config(page_title="Smart Notes (Semantic Search)", layout="wide")
st.title("🗒️ Smart Notes — Semantic Search (SBERT)")
st.caption("Search by meaning, not keywords. Model: all-MiniLM-L6-v2")

DATA_PATH = Path(__file__).resolve().parent / "data" / "notes.json"
EMB_PATH = Path(__file__).resolve().parent / "models" / "note_embeddings.npy"

if "notes" not in st.session_state:
    st.session_state.notes = to_uniform_notes(load_notes(DATA_PATH))
if "note_embs" not in st.session_state:
    st.session_state.note_embs = None
if "last_hash" not in st.session_state:
    st.session_state.last_hash = ""

def notes_hash(notes):
    payload = json.dumps(notes, sort_keys=True, ensure_ascii=False).encode("utf-8")
    import hashlib
    return hashlib.sha256(payload).hexdigest()

with st.sidebar:
    st.header("📚 Notes Manager")
    with st.expander("➕ Add a Note"):
        content = st.text_area("Content", height=120)
        col1, col2 = st.columns(2)
        with col1:
            category = st.text_input("Category", value="General")
        with col2:
            tags_csv = st.text_input("Tags (comma-separated)", value="")
        if st.button("Add Note", type="primary"):
            if content.strip():
                tags = [t.strip() for t in tags_csv.split(",") if t.strip()]
                st.session_state.notes.append({"content": content.strip(), "category": category.strip() or "General", "tags": tags})
                st.success("Note added.")
            else:
                st.warning("Please enter some content.")

    with st.expander("🪓 Chunk a Long Note"):
        long_note = st.text_area("Paste long note", height=120)
        max_words = st.slider("Max words per chunk", 10, 120, 50, 5)
        if st.button("Create Chunks"):
            if long_note.strip():
                chunks = chunk_note(long_note, max_words=max_words)
                for ch in chunks:
                    st.session_state.notes.append({"content": ch, "category": "Chunked", "tags": ["chunk"]})
                st.success(f"Added {len(chunks)} chunks.")
            else:
                st.warning("Please paste a note to chunk.")

    with st.expander("📥 Import / 📤 Export"):
        uploaded = st.file_uploader("Import notes (JSON list)", type=["json"])
        if uploaded:
            try:
                data = json.load(uploaded)
                st.session_state.notes.extend(to_uniform_notes(data))
                st.success(f"Imported {len(data)} notes.")
            except Exception as e:
                st.error(f"Import failed: {e}")
        if st.button("Export notes.json"):
            st.download_button(
                "Download notes.json",
                data=json.dumps(st.session_state.notes, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="notes.json",
                mime="application/json",
                use_container_width=True
            )

    if st.button("💾 Save notes to disk"):
        save_notes(st.session_state.notes, DATA_PATH)
        st.success(f"Saved to {DATA_PATH}")

current_hash = notes_hash(st.session_state.notes)
if current_hash != st.session_state.last_hash:
    st.session_state.note_embs = None

colA, colB = st.columns([2, 1])
with colA:
    st.subheader("🔎 Search")
    query = st.text_input("Type a question or concept", placeholder="e.g., How do Python lists work?")
    top_k = st.slider("Top K results", 1, 20, 5)
    min_sim = st.slider("Minimum similarity", 0.0, 1.0, 0.0, 0.01)
with colB:
    st.subheader("Filters")
    categories = sorted(set(n.get("category", "General") for n in st.session_state.notes))
    selected_cat = st.selectbox("Category", options=["(any)"] + categories)
    tags_filter = st.text_input("Required tags (comma-separated)", value="")
    req_tags = [t.strip() for t in tags_filter.split(",") if t.strip()]

if query:
    if st.session_state.note_embs is None:
        with st.spinner("Encoding notes (downloads model on first run)..."):
            st.session_state.note_embs = compute_embeddings(st.session_state.notes)
            st.session_state.last_hash = current_hash

    category_param = None if selected_cat == "(any)" else selected_cat
    results = cosine_search(
        query=query,
        notes=st.session_state.notes,
        note_embs=st.session_state.note_embs,
        top_k=top_k,
        min_similarity=min_sim,
        category=category_param,
        required_tags=req_tags if req_tags else None
    )

    st.write(f"**Found {len(results)} result(s).**")
    for i, r in enumerate(results, 1):
        n = r['note']
        with st.container(border=True):
            st.markdown(f"**{i}. Similarity:** `{r['similarity']:.3f}`  "
                        f"**Category:** `{n.get('category','General')}`  — "
                        f"**Tags:** `{', '.join(n.get('tags', [])) or '(none)'}`")
            st.write(n.get('content','').strip())
else:
    st.info("Enter a search query to begin. Add notes from the sidebar.")

st.markdown("---")
st.subheader("📄 All Notes")
st.dataframe(
    [{"content": n.get("content",""), "category": n.get("category","General"),
        "tags": ", ".join(n.get("tags", []))} for n in st.session_state.notes],
    use_container_width=True, height=300
)

col1, col2 = st.columns(2)
with col1:
    if st.button("💽 Save embeddings (.npy)"):
        if st.session_state.note_embs is None:
            st.warning("No embeddings yet. Run a search first.")
        else:
            np.save(EMB_PATH, st.session_state.note_embs)
            st.success(f"Saved: {EMB_PATH}")
with col2:
    if st.button("📂 Load embeddings (.npy)"):
        if EMB_PATH.exists():
            st.session_state.note_embs = np.load(EMB_PATH)
            st.success(f"Loaded: {EMB_PATH}")
        else:
            st.warning("No saved embeddings found.")
