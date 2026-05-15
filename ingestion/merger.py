import os
import json
from pathlib import Path
from ingestion.pdf_extractor import extract_smart
from ingestion.chunker import chunk_pages
from ingestion.vision_extractor import (
    pdf_page_to_base64,
    describe_page_with_vision,
    extract_images_from_pdf
)


def vision_description_to_text(vision_data: dict) -> str:
    """
    Convert vision API response dict into readable text.
    This text gets chunked and embedded alongside regular text.
    """
    parts = []

    if vision_data.get("summary"):
        parts.append(f"Visual Summary: {vision_data['summary']}")

    if vision_data.get("charts"):
        for chart in vision_data["charts"]:
            parts.append(f"Chart: {chart}")

    if vision_data.get("tables"):
        for table in vision_data["tables"]:
            parts.append(f"Table: {table}")

    if vision_data.get("diagrams"):
        for diagram in vision_data["diagrams"]:
            parts.append(f"Diagram: {diagram}")

    if vision_data.get("key_text"):
        for text in vision_data["key_text"]:
            parts.append(f"Visual Text: {text}")

    return "\n".join(parts)


def process_pdf_with_vision(
    pdf_path: str,
    use_vision: bool = True,
    max_vision_pages: int = 5
) -> dict:
    """
    Complete multimodal processing of a PDF.

    Steps:
    1. Extract text normally
    2. Chunk the text
    3. Run vision on each page (if use_vision=True)
    4. Merge everything into one document object

    Args:
        pdf_path: Path to PDF
        use_vision: Whether to use GPT-4o Vision (costs API credits)
        max_vision_pages: Max pages to run vision on (cost control)

    Returns:
        Complete document dict with text chunks + vision data
    """
    print(f"\nProcessing: {os.path.basename(pdf_path)}")
    print("-" * 40)

    # Step 1: Extract text
    print("Step 1: Extracting text...")
    pages = extract_smart(pdf_path)

    # Step 2: Chunk text
    print("Step 2: Chunking text...")
    text_chunks = chunk_pages(pages)
    print(f"  Created {len(text_chunks)} text chunks")

    # Step 3: Vision analysis
    vision_results = []
    if use_vision:
        print(f"Step 3: Running vision on up to "
              f"{max_vision_pages} pages...")

        pages_to_analyze = min(len(pages), max_vision_pages)

        for page_num in range(pages_to_analyze):
            print(f"  Analyzing page {page_num + 1}/{pages_to_analyze}...")
            try:
                b64 = pdf_page_to_base64(pdf_path, page_num)
                vision_data = describe_page_with_vision(b64, page_num)
                vision_results.append(vision_data)

                # Show what was found
                found = []
                if vision_data.get("has_charts"):
                    found.append("charts")
                if vision_data.get("has_tables"):
                    found.append("tables")
                if vision_data.get("has_diagrams"):
                    found.append("diagrams")

                if found:
                    print(f"    Found: {', '.join(found)}")
                else:
                    print(f"    No visual elements found")

            except Exception as e:
                print(f"  Vision failed on page {page_num + 1}: {e}")
    else:
        print("Step 3: Skipping vision (use_vision=False)")

    # Step 4: Extract embedded images
    print("Step 4: Extracting embedded images...")
    embedded_images = extract_images_from_pdf(pdf_path)

    # Step 5: Create vision text chunks
    vision_chunks = []
    for v in vision_results:
        vision_text = vision_description_to_text(v)
        if vision_text.strip():
            vision_chunks.append({
                "chunk_id": f"vision_p{v['page']}",
                "text": vision_text,
                "source": os.path.basename(pdf_path),
                "page": v["page"],
                "char_count": len(vision_text),
                "chunk_type": "vision"
            })

    # Mark regular chunks
    for chunk in text_chunks:
        chunk["chunk_type"] = "text"

    # Combine all chunks
    all_chunks = text_chunks + vision_chunks

    # Build final document object
    document = {
        "source": os.path.basename(pdf_path),
        "full_path": str(pdf_path),
        "total_pages": len(pages),
        "text_chunks": len(text_chunks),
        "vision_chunks": len(vision_chunks),
        "total_chunks": len(all_chunks),
        "embedded_images": len(embedded_images),
        "chunks": all_chunks,
        "vision_data": vision_results
    }

    print(f"\nResult:")
    print(f"  Text chunks:   {len(text_chunks)}")
    print(f"  Vision chunks: {len(vision_chunks)}")
    print(f"  Total chunks:  {len(all_chunks)}")
    print(f"  Images found:  {len(embedded_images)}")

    return document


def save_document(document: dict, output_dir: str = "data/processed"):
    """Save processed document to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    # remove base64 image data before saving (too large)
    doc_to_save = document.copy()
    doc_to_save["vision_data"] = [
        {k: v for k, v in vd.items() if k != "base64"}
        for vd in document.get("vision_data", [])
    ]

    filename = Path(document["source"]).stem
    output_path = f"{output_dir}/{filename}_processed.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(doc_to_save, f, indent=2, ensure_ascii=False)

    print(f"Saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    import sys

    # Test without vision first (no API cost)
    test_pdf = "data/raw/sample3.pdf"

    if not os.path.exists(test_pdf):
        print(f"Test PDF not found at {test_pdf}")
        print("Run: python scripts/download_samples.py")
    else:
        # Test WITHOUT vision first
        print("Testing without vision (free)...")
        doc = process_pdf_with_vision(
            test_pdf,
            use_vision=False
        )
        output = save_document(doc)
        print(f"\nSuccess! Document saved to: {output}")