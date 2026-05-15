import os
import sys
import json
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))


def check_results():
    output_dir = "data/processed"
    summary_path = f"{output_dir}/summary.json"

    # Load summary
    if not os.path.exists(summary_path):
        print("No summary found. Run test_batch.py first.")
        return

    with open(summary_path, encoding='utf-8') as f:
        data = json.load(f)

    summary = data["summary"]
    results = data["results"]

    print("=" * 50)
    print("DATASET SUMMARY")
    print("=" * 50)
    print(f"Total PDFs processed: {summary['total_files']}")
    print(f"Successful:           {summary['successful']}")
    print(f"Errors:               {summary['errors']}")
    print(f"Total chunks:         {summary['total_chunks']}")
    print(f"Avg chunks per doc:   "
          f"{summary['total_chunks'] // max(summary['successful'], 1)}")

    # Show errors if any
    errors = [r for r in results if r["status"] == "error"]
    if errors:
        print(f"\nFailed files ({len(errors)}):")
        for e in errors:
            print(f"  - {e['file']}: {e.get('error', 'unknown')[:60]}")

    # Show top 5 largest docs
    success = [r for r in results if r["status"] == "success"]
    success.sort(key=lambda x: x.get("chunks", 0), reverse=True)

    print(f"\nTop 5 largest documents (by chunks):")
    for r in success[:5]:
        print(f"  {r['file'][:40]:40} "
              f"{r.get('pages', 0):3} pages  "
              f"{r.get('chunks', 0):4} chunks")

    # Chunk size distribution
    print(f"\nChunk distribution:")
    small  = len([r for r in success if r.get("chunks", 0) < 20])
    medium = len([r for r in success if 20 <= r.get("chunks", 0) < 100])
    large  = len([r for r in success if r.get("chunks", 0) >= 100])
    print(f"  Small  (<20 chunks):   {small} docs")
    print(f"  Medium (20-100):       {medium} docs")
    print(f"  Large  (100+ chunks):  {large} docs")

    processed = list(Path(output_dir).glob("*_processed.json"))
    print(f"\nProcessed JSON files on disk: {len(processed)}")


def check_single_doc(filename):
    """Check one processed document in detail."""
    path = f"data/processed/{filename}_processed.json"

    if not os.path.exists(path):
        print(f"File not found: {path}")
        # show available files
        available = list(Path("data/processed").glob("*_processed.json"))
        if available:
            print("Available files:")
            for f in available[:5]:
                print(f"  {f.stem}")
        return

    with open(path, encoding='utf-8') as f:
        doc = json.load(f)

    print(f"\nDocument: {doc['source']}")
    print(f"Pages:    {doc['total_pages']}")
    print(f"Chunks:   {doc['total_chunks']}")
    print(f"\nFirst 3 chunks:")
    for chunk in doc["chunks"][:3]:
        print(f"\n  Chunk {chunk['chunk_id']} "
              f"(page {chunk['page']}, "
              f"{chunk['char_count']} chars):")
        print(f"  {chunk['text'][:150]}...")


if __name__ == "__main__":
    check_results()

    # Check one document in detail
    print("\n" + "=" * 50)
    print("SAMPLE DOCUMENT DETAIL")
    print("=" * 50)
    check_single_doc("rag_original")