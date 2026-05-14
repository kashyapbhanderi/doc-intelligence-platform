from paddleocr import PaddleOCR
import fitz  # PyMuPDF
import os
import tempfile


# Initialize OCR engine once (slow to load, so we do it once)
ocr_engine = None

def get_ocr_engine():
    """
    Load OCR engine once and reuse it.
    First call takes ~10 seconds to load the model.
    """
    global ocr_engine
    if ocr_engine is None:
        print("Loading OCR engine for first time (this takes ~10 seconds)...")
        ocr_engine = PaddleOCR(
            use_textline_orientation=True,  # detect rotated text
            lang='en',           # English language
            # show_log=False       # hide verbose logs
        )
        print("OCR engine loaded!")
    return ocr_engine


def extract_text_with_ocr(pdf_path: str) -> list:
    """
    Extract text from a scanned PDF using OCR.
    Converts each page to an image, then reads the image.
    Returns same format as pdf_extractor.py for compatibility.
    """
    ocr = get_ocr_engine()
    doc = fitz.open(pdf_path)
    pages = []

    print(f"Running OCR on {len(doc)} pages...")

    for page_num in range(len(doc)):
        page = doc[page_num]

        # Convert PDF page to image (300 DPI for good quality)
        mat = fitz.Matrix(300/72, 300/72)
        pix = page.get_pixmap(matrix=mat)

        # Save image to temp file
        with tempfile.NamedTemporaryFile(
            suffix='.png', delete=False
        ) as tmp:
            tmp_path = tmp.name
            pix.save(tmp_path)

        # Run OCR on the image
        result = ocr.ocr(tmp_path, cls=True)

        # Extract text from OCR result
        page_text = ""
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text_info = line[1]
                    if text_info and len(text_info) >= 1:
                        page_text += text_info[0] + " "

        # Clean up temp file
        os.unlink(tmp_path)

        pages.append({
            "page": page_num + 1,
            "text": page_text.strip(),
            "source": os.path.basename(pdf_path),
            "total_pages": len(doc),
            "extraction_method": "ocr"
        })

        print(f"  Page {page_num + 1}/{len(doc)} done "
              f"({len(page_text)} chars extracted)")

    doc.close()
    return pages


if __name__ == "__main__":
    print("OCR Extractor ready.")
    print("Usage: extract_text_with_ocr('path/to/scanned.pdf')")
    # Load engine to verify it works
    get_ocr_engine()
    print("All good!")