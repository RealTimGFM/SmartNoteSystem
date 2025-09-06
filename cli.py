
import argparse, json
from pathlib import Path
from smartnotes.storage import load_notes
from smartnotes.search import to_uniform_notes, compute_embeddings, cosine_search

def main():
    ap = argparse.ArgumentParser(description="Semantic search over notes (CLI)")
    ap.add_argument("--query", required=True, help="Your search query")
    ap.add_argument("--top_k", type=int, default=5, help="Number of results")
    ap.add_argument("--min_similarity", type=float, default=0.0, help="Min cosine similarity (0..1)")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent / "data" / "notes.json"))
    args = ap.parse_args()

    notes = to_uniform_notes(load_notes(args.data))
    embs = compute_embeddings(notes)
    results = cosine_search(args.query, notes, embs, top_k=args.top_k, min_similarity=args.min_similarity)
    for i, r in enumerate(results, 1):
        print(f"\n{i}. sim={r['similarity']:.3f} | cat={r['note'].get('category')} | tags={','.join(r['note'].get('tags', []))}")
        print(r['note']['content'])

if __name__ == "__main__":
    main()
