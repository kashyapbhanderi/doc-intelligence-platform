import os
import json
from pathlib import Path
from ingestion.pdf_extractor import extract_smart
from ingestion.chunker import chunk_pages, save_chunks


def process_single_pdf(pdf_path: str, output_dir: str = "data/processed") -> dict:
    """
    Full pipeline for one PDF:
    1. Extract text (smart - normal or OCR)
    2. Chunk the text
    3. Save to JSON
    Returns summary stats.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Extract
    pages = extract_smart(pdf_path)

    if not pages:
        print(f"  WARNING: No text extracted from {pdf_path}")
        return {"file": pdf_path, "status": "failed", "chunks": 0}

    # Step 2: Chunk
    chunks = chunk_pages(pages)

    # Step 3: Save
    filename = Path(pdf_path).stem  # filename without extension
    output_path = f"{output_dir}/{filename}_chunks.json"
    save_chunks(chunks, output_path)

    summary = {
        "file": os.path.basename(pdf_path),
        "status": "success",
        "pages": len(pages),
        "chunks": len(chunks),
        "output": output_path
    }

    print(f"  Done: {len(pages)} pages → {len(chunks)} chunks")
    return summary


def process_all_pdfs(input_dir: str = "data/raw",
                     output_dir: str = "data/processed") -> list:
    """
    Process ALL PDFs in a folder.
    Shows progress and saves results summary.
    """
    from tqdm import tqdm

    # Find all PDFs
    pdf_files = list(Path(input_dir).glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}/")
        print("Add some PDF files to data/raw/ folder first.")
        return []

    print(f"Found {len(pdf_files)} PDFs to process...")
    results = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        result = process_single_pdf(str(pdf_path), output_dir)
        results.append(result)

    # Save summary
    summary_path = f"{output_dir}/pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print final stats
    successful = [r for r in results if r["status"] == "success"]
    total_chunks = sum(r["chunks"] for r in successful)

    print(f"\n{'='*40}")
    print(f"Pipeline complete!")
    print(f"Processed: {len(successful)}/{len(pdf_files)} PDFs")
    print(f"Total chunks: {total_chunks}")
    print(f"Summary saved: {summary_path}")

    return results


if __name__ == "__main__":
    # Test with a single PDF
    # First download any PDF and put it in data/raw/
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    print("Pipeline ready!")
    print("Add PDFs to data/raw/ folder then run process_all_pdfs()")