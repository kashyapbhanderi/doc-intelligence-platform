import fitz  # PyMuPDF
import os
from pathlib import Path

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