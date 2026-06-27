"""
scripts/build_graph.py
========================
One-time script to build the knowledge graph from your ingested chunks.

Run ONCE after Week 2 ingestion is complete:
    python scripts/build_graph.py

Then the graph is saved to data/knowledge_graph.pkl and loaded
automatically by HybridGraphRAG on every agent query.

WHERE TO PLACE THIS FILE: scripts/build_graph.py
"""

import json
import sys
import time
from pathlib import Path

# ── Add project root to path ──────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from knowledge_graph.graph_builder import GraphBuilder


def load_chunks(chunks_path: str = "data/processed") -> list[dict]:
    """
    Load chunks either from:
      - a directory of *_processed.json files (your actual ingestion output), or
      - a single combined JSON file (list or {"chunks": [...]})
    """
    path = Path(chunks_path)

    if path.is_dir():
        json_files = sorted(path.glob("*_processed.json"))
        if not json_files:
            print(f"ERROR: no *_processed.json files found in {chunks_path}")
            print("Make sure you have run embeddings/ingest.py first (Week 2).")
            sys.exit(1)

        chunks = []
        for jf in json_files:
            with open(jf, encoding="utf-8") as f:
                doc = json.load(f)
            chunks.extend(doc.get("chunks", []))

        print(f"Aggregated {len(chunks)} chunks from {len(json_files)} processed files")

    elif path.is_file():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            chunks = data
        elif isinstance(data, dict) and "chunks" in data:
            chunks = data["chunks"]
        else:
            print(f"ERROR: unexpected format in {chunks_path}")
            sys.exit(1)

    else:
        print(f"ERROR: path not found: {chunks_path}")
        print("Make sure you have run embeddings/ingest.py first (Week 2).")
        sys.exit(1)

    # Normalise keys to what GraphBuilder expects
    normalised = []
    for i, chunk in enumerate(chunks):
        normalised.append({
            "id":     chunk.get("id")     or chunk.get("chunk_id") or str(i),
            "text":   chunk.get("text")   or chunk.get("content")  or "",
            "source": chunk.get("source") or chunk.get("filename") or chunk.get("file", ""),
            "page":   int(chunk.get("page") or chunk.get("page_num") or 0),
        })

    return normalised

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build GraphRAG knowledge graph")
    parser.add_argument("--chunks", default="data/processed", help="Path to processed chunks (directory of *_processed.json, or a single combined JSON file)")
    parser.add_argument("--output",    default="data/knowledge_graph.pkl", help="Output path for graph")
    parser.add_argument("--force",     action="store_true",               help="Rebuild even if graph exists")
    args = parser.parse_args()

    # Skip rebuild if graph already exists (unless --force)
    if Path(args.output).exists() and not args.force:
        print(f"Graph already exists at {args.output}.")
        print("Loading and showing stats...")
        gb = GraphBuilder(graph_path=args.output)
        gb.load()
        stats = gb.get_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")
        print("\nTo rebuild from scratch, use: python scripts/build_graph.py --force")
        return

    print("=" * 60)
    print("GraphRAG — Knowledge Graph Builder")
    print("=" * 60)

    # 1. Load chunks
    print(f"\nLoading chunks from {args.chunks}...")
    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks")

    # 2. Build graph
    print("\nBuilding graph (this may take 5-15 minutes for 500+ documents)...")
    print("Entity extraction: spaCy (fast)\nRelationship extraction: GPT-4o-mini (API calls for chunks with 2+ entities)")
    print("-" * 60)

    gb = GraphBuilder(graph_path=args.output)
    t0 = time.time()
    gb.build_from_chunks(chunks)
    elapsed = time.time() - t0

    # 3. Show stats
    stats = gb.get_stats()
    print("\n" + "=" * 60)
    print("GRAPH BUILD COMPLETE")
    print("=" * 60)
    print(f"  Time taken    : {elapsed:.1f}s")
    for k, v in stats.items():
        print(f"  {k:<22}: {v}")
    print(f"\nGraph saved to: {args.output}")
    print("\nNext step: your HybridGraphRAG will now load this graph automatically.")
    print("Verify with: python scripts/test_graphrag_query.py")


if __name__ == "__main__":
    main()
