"""
agents/editor_tools/pdf_tools.py
Professional PDF manipulation tools using PyMuPDF.

Features:
- PDF metadata extraction
- Watermarking
- Merge PDFs
- Split PDFs
- PDF → DOCX conversion

Author: Kashyap
Project: Doc Intelligence Platform
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional

import fitz  # PyMuPDF


# ---------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# PDF INFO
# ---------------------------------------------------------

def get_pdf_info(path: str) -> Dict:
    """
    Extract metadata and preview text from a PDF.

    Args:
        path: Path to PDF file

    Returns:
        Dictionary containing PDF information
    """

    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}

    try:
        doc = fitz.open(path)

        metadata = doc.metadata
        pages = len(doc)

        preview = ""
        if pages > 0:
            preview = doc[0].get_text()[:200]

        file_size_kb = os.path.getsize(path) // 1024

        result = {
            "filename": Path(path).name,
            "pages": pages,
            "file_size": f"{file_size_kb} KB",
            "title": metadata.get("title", ""),
            "author": metadata.get("author", ""),
            "preview": preview.strip()
        }

        doc.close()

        return result

    except Exception as e:
        logger.error(f"PDF info extraction failed: {e}")
        return {"error": str(e)}


# ---------------------------------------------------------
# WATERMARK PDF
# ---------------------------------------------------------

def watermark_pdf(
    path: str,
    text: str = "CONFIDENTIAL",
    output_path: Optional[str] = None
) -> str:
    """
    Add watermark text to all pages of a PDF.

    Args:
        path: Input PDF path
        text: Watermark text
        output_path: Output PDF path

    Returns:
        Success/error message
    """

    if not os.path.exists(path):
        return f"Error: File not found -> {path}"

    try:
        if output_path is None:
            stem = Path(path).stem
            output_path = str(
                Path(path).parent /
                f"{stem}_watermarked.pdf"
            )

        doc = fitz.open(path)

        for page in doc:

            width = page.rect.width
            height = page.rect.height

            page.insert_text(
                fitz.Point(width * 0.2, height * 0.5),
                text,
                fontsize=40,
                color=(1, 0, 0),
                rotate=0,
                overlay=True
            )

        doc.save(output_path)
        doc.close()

        logger.info("Watermark added successfully")

        return f"Watermarked PDF saved -> {output_path}"

    except Exception as e:
        logger.error(f"Watermark failed: {e}")
        return f"Error: {e}"


# ---------------------------------------------------------
# MERGE PDFS
# ---------------------------------------------------------

def merge_pdfs(
    paths: List[str],
    output_path: str = "data/test_docs/merged.pdf"
) -> str:
    """
    Merge multiple PDFs into one document.

    Args:
        paths: List of PDF paths
        output_path: Output merged PDF path

    Returns:
        Success/error message
    """

    missing = [p for p in paths if not os.path.exists(p)]

    if missing:
        return f"Error: Missing files -> {missing}"

    try:
        merged_doc = fitz.open()

        total_pages = 0

        for path in paths:

            doc = fitz.open(path)

            merged_doc.insert_pdf(doc)

            total_pages += len(doc)

            doc.close()

        Path(output_path).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        merged_doc.save(output_path)
        merged_doc.close()

        logger.info("PDF merge completed")

        return (
            f"Merged {len(paths)} PDFs "
            f"({total_pages} pages) -> {output_path}"
        )

    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return f"Error: {e}"


# ---------------------------------------------------------
# SPLIT PDF
# ---------------------------------------------------------

def split_pdf(
    path: str,
    start_page: int = 1,
    end_page: Optional[int] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Extract specific pages from a PDF.

    Args:
        path: Input PDF path
        start_page: Starting page (1-indexed)
        end_page: Ending page (1-indexed)
        output_path: Output PDF path

    Returns:
        Success/error message
    """

    if not os.path.exists(path):
        return f"Error: File not found -> {path}"

    try:
        doc = fitz.open(path)

        total_pages = len(doc)

        start = max(0, start_page - 1)

        end = end_page if end_page else total_pages

        end = min(end, total_pages)

        if start >= end:
            return (
                f"Error: Invalid page range "
                f"{start_page}-{end_page}"
            )

        if output_path is None:
            stem = Path(path).stem

            output_path = str(
                Path(path).parent /
                f"{stem}_split.pdf"
            )

        new_doc = fitz.open()

        new_doc.insert_pdf(
            doc,
            from_page=start,
            to_page=end - 1
        )

        new_doc.save(output_path)

        new_doc.close()
        doc.close()

        logger.info("PDF split completed")

        return (
            f"Extracted pages "
            f"{start_page}-{end} -> {output_path}"
        )

    except Exception as e:
        logger.error(f"Split failed: {e}")
        return f"Error: {e}"


# ---------------------------------------------------------
# PDF TO DOCX
# ---------------------------------------------------------

def pdf_to_docx(
    path: str,
    output_path: Optional[str] = None
) -> str:
    """
    Convert PDF to DOCX format.

    Args:
        path: Input PDF path
        output_path: Output DOCX path

    Returns:
        Success/error message
    """

    if not os.path.exists(path):
        return f"Error: File not found -> {path}"

    try:
        from pdf2docx import Converter

        if output_path is None:
            output_path = str(
                Path(path).with_suffix(".docx")
            )

        converter = Converter(path)

        converter.convert(
            output_path,
            start=0,
            end=None
        )

        converter.close()

        size_kb = os.path.getsize(output_path) // 1024

        logger.info("PDF converted to DOCX")

        return (
            f"Converted successfully -> "
            f"{Path(output_path).name} "
            f"({size_kb} KB)"
        )

    except ImportError:
        return (
            "pdf2docx is not installed.\n"
            "Run: pip install pdf2docx"
        )

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        return f"Error: {e}"


# ---------------------------------------------------------
# MAIN TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    import shutil

    RAW_DIR = Path("data/raw")
    TEST_DIR = Path("data/test_docs")

    TEST_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_pdfs = list(RAW_DIR.glob("*.pdf"))

    if not raw_pdfs:
        print("No PDFs found in data/raw")
        exit()

    TEST_PDF = raw_pdfs[0]

    copied_pdf = TEST_DIR / "test_document.pdf"

    shutil.copy(TEST_PDF, copied_pdf)

    print("=" * 60)
    print("TESTING PDF TOOLS")
    print("=" * 60)

    # PDF INFO
    info = get_pdf_info(str(copied_pdf))
    print("\nPDF INFO")
    print(info)

    # WATERMARK
    print("\nWATERMARK TEST")
    print(
        watermark_pdf(
            str(copied_pdf),
            "DRAFT",
            str(TEST_DIR / "watermarked.pdf")
        )
    )

    # SPLIT
    print("\nSPLIT TEST")
    print(
        split_pdf(
            str(copied_pdf),
            start_page=1,
            end_page=2,
            output_path=str(TEST_DIR / "split.pdf")
        )
    )

    # MERGE
    print("\nMERGE TEST")
    print(
        merge_pdfs(
            [str(copied_pdf), str(copied_pdf)],
            str(TEST_DIR / "merged.pdf")
        )
    )

    print("\nALL PDF TOOLS WORKING SUCCESSFULLY")