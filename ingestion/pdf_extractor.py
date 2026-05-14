import fitz  # PyMuPDF
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

def extract_text_from_pdf(pdf_path: str) -> list:
    """
    Extract text from each page of a PDF file.
    Returns a list of dicts with page number and text.
    """
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        pages.append({
            "page": page_num + 1,
            "text": text.strip(),
            "source": os.path.basename(pdf_path),
            "total_pages": len(doc)
        })
    
    doc.close()
    return pages

def is_scanned_pdf(pages: list) -> bool:
    """
    Check if PDF is scanned (image-based).
    If average text per page is less than 50 chars, it is scanned.
    """
    avg_text = sum(len(p["text"]) for p in pages) / len(pages)
    return avg_text < 50


if __name__ == "__main__":
    # Quick test — create a sample PDF path
    test_path = "data/sample.pdf"
    
    if not os.path.exists(test_path):
        print("No sample PDF found. Download any PDF and put it in data/ folder.")
        print("Rename it to sample.pdf")
    else:
        print("Extracting text from:", test_path)
        pages = extract_text_from_pdf(test_path)
        print(f"Total pages: {len(pages)}")
        print(f"Is scanned: {is_scanned_pdf(pages)}")
        print(f"\nFirst 200 chars of page 1:")
        print(pages[0]["text"][:200])

def extract_with_fallback(pdf_path: str) -> list:
    """
    Smart extraction with automatic fallback.
    
    Logic:
    1. Try normal PyMuPDF extraction first (fast)
    2. Check if extracted text is too short (scanned PDF)
    3. If scanned, fall back to OCR automatically
    
    This handles ALL PDF types with one function call.
    """
    print(f"\nProcessing: {os.path.basename(pdf_path)}")

    # Step 1: Try normal extraction
    pages = extract_text_from_pdf(pdf_path)

    # Step 2: Check if it is a scanned PDF
    if is_scanned_pdf(pages):
        print("  Detected: Scanned PDF — switching to OCR...")
        # Import here to avoid loading OCR if not needed
        from ingestion.ocr_extractor import extract_text_with_ocr
        pages = extract_text_with_ocr(pdf_path)
        print("  OCR complete!")
    else:
        print(f"  Detected: Digital PDF — extracted "
              f"{sum(len(p['text']) for p in pages)} characters")
        # Mark extraction method
        for p in pages:
            p["extraction_method"] = "pymupdf"

    return pages

def process_pdf_to_chunks(pdf_path: str,
                           chunk_size: int = 512,
                           chunk_overlap: int = 50) -> list:
    """
    Full pipeline: PDF file → clean text chunks.
    This is the main function you call from other parts of the project.
    """
    from ingestion.chunker import chunk_pages

    # Extract text
    pages = extract_with_fallback(pdf_path)

    # Remove pages with no text
    pages = [p for p in pages if len(p["text"].strip()) > 10]
    print(f"  Valid pages: {len(pages)}")

    # Chunk the text
    chunks = chunk_pages(pages, chunk_size, chunk_overlap)
    print(f"  Chunks created: {len(chunks)}")

    return chunks    

def extract_smart(pdf_path: str) -> list:
    print(f"Extracting: {os.path.basename(pdf_path)}")
    pages = extract_text_from_pdf(pdf_path)
    if is_scanned_pdf(pages):
        print("  Detected scanned PDF. Switching to OCR...")
        from ingestion.ocr_extractor import extract_text_with_ocr
        pages = extract_text_with_ocr(pdf_path)
        print(f"  OCR complete. Extracted {len(pages)} pages.")
    else:
        print(f"  Normal extraction successful. {len(pages)} pages.")
    return pages