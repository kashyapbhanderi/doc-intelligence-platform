import os
import json
from pathlib import Path
from ingestion.pdf_extractor import extract_smart
from ingestion.chunker import chunk_pages


def vision_description_to_text(vision_data: dict) -> str:
    """
    Convert vision API response dict into readable text.
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
    use_vision: bool = False,
    max_vision_pages: int = 0
) -> dict:
    """
    Complete multimodal processing of a PDF.
    Steps:
    1. Extract text
    2. Chunk the text
    3. Run vision on pages (only if use_vision=True)
    4. Merge into one document object
    """
    print(f"\nProcessing: {os.path.basename(pdf_path)}")

    # Step 1: Extract text
    pages = extract_smart(pdf_path)

    if not pages:
        return {
            "source": os.path.basename(pdf_path),
            "full_path": str(pdf_path),
            "total_pages": 0,
            "text_chunks": 0,
            "vision_chunks": 0,
            "total_chunks": 0,
            "embedded_images": 0,
            "chunks": [],
            "vision_data": []
        }

    # Step 2: Chunk text
    text_chunks = chunk_pages(pages)

    # Mark regular chunks
    for chunk in text_chunks:
        chunk["chunk_type"] = "text"

    # Step 3: Vision analysis (skip if use_vision=False)
    vision_results = []
    vision_chunks = []

    if use_vision and max_vision_pages > 0:
        try:
            from ingestion.vision_extractor import (
                pdf_page_to_base64,
                describe_page_with_vision,
                extract_images_from_pdf
            )

            pages_to_analyze = min(len(pages), max_vision_pages)
            print(f"  Running vision on {pages_to_analyze} pages...")

            for page_num in range(pages_to_analyze):
                try:
                    b64 = pdf_page_to_base64(pdf_path, page_num)
                    vision_data = describe_page_with_vision(b64, page_num)
                    vision_results.append(vision_data)

                    vision_text = vision_description_to_text(vision_data)
                    if vision_text.strip():
                        vision_chunks.append({
                            "chunk_id": f"vision_p{vision_data['page']}",
                            "text": vision_text,
                            "source": os.path.basename(pdf_path),
                            "page": vision_data["page"],
                            "char_count": len(vision_text),
                            "chunk_type": "vision"
                        })
                except Exception as e:
                    print(f"  Vision failed page {page_num + 1}: {e}")

        except ImportError:
            print("  Vision extractor not available, skipping...")

    # Step 4: Try to get embedded images count
    embedded_count = 0
    try:
        from ingestion.vision_extractor import extract_images_from_pdf
        embedded_images = extract_images_from_pdf(pdf_path)
        embedded_count = len(embedded_images)
    except Exception:
        pass

    # Combine all chunks
    all_chunks = text_chunks + vision_chunks

    document = {
        "source": os.path.basename(pdf_path),
        "full_path": str(pdf_path),
        "total_pages": len(pages),
        "text_chunks": len(text_chunks),
        "vision_chunks": len(vision_chunks),
        "total_chunks": len(all_chunks),
        "embedded_images": embedded_count,
        "chunks": all_chunks,
        "vision_data": vision_results
    }

    print(f"  Done: {len(pages)} pages → {len(all_chunks)} chunks")
    return document


def save_document(document: dict, output_dir: str = "data/processed"):
    """Save processed document to JSON file."""
    os.makedirs(output_dir, exist_ok=True)

    filename = Path(document["source"]).stem
    output_path = f"{output_dir}/{filename}_processed.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document, f, indent=2, ensure_ascii=False)

    return output_path


if __name__ == "__main__":
    test_pdf = "data/raw/rag_original.pdf"

    if not os.path.exists(test_pdf):
        # try any PDF in data/raw
        pdfs = list(Path("data/raw").glob("*.pdf"))
        if pdfs:
            test_pdf = str(pdfs[0])
        else:
            print("No PDFs found in data/raw/")
            exit()

    doc = process_pdf_with_vision(test_pdf, use_vision=False)
    output = save_document(doc)
    print(f"\nSaved to: {output}")
    print(f"Total chunks: {doc['total_chunks']}")