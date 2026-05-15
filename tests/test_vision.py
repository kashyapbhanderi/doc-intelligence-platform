import pytest
import sys
import os
import json
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from ingestion.vision_extractor import (
    pdf_page_to_base64,
    extract_images_from_pdf,
    vision_description_to_text
)
from ingestion.merger import vision_description_to_text


def test_vision_description_to_text_with_charts():
    """Vision data with charts should produce text mentioning Chart."""
    vision_data = {
        "has_charts": True,
        "charts": ["Bar chart showing revenue from 2020 to 2024"],
        "has_tables": False,
        "tables": [],
        "has_diagrams": False,
        "diagrams": [],
        "key_text": [],
        "summary": "Page with revenue chart"
    }
    text = vision_description_to_text(vision_data)
    assert "Chart" in text
    assert "revenue" in text.lower()


def test_vision_description_to_text_with_tables():
    """Vision data with tables should produce text mentioning Table."""
    vision_data = {
        "has_charts": False,
        "charts": [],
        "has_tables": True,
        "tables": ["Name | Age | City"],
        "has_diagrams": False,
        "diagrams": [],
        "key_text": [],
        "summary": "Page with data table"
    }
    text = vision_description_to_text(vision_data)
    assert "Table" in text


def test_vision_description_empty_data():
    """Empty vision data should return empty string."""
    vision_data = {
        "has_charts": False,
        "charts": [],
        "has_tables": False,
        "tables": [],
        "has_diagrams": False,
        "diagrams": [],
        "key_text": [],
        "summary": ""
    }
    text = vision_description_to_text(vision_data)
    assert text.strip() == ""


def test_vision_description_with_summary():
    """Summary should always be included in output."""
    vision_data = {
        "has_charts": False,
        "charts": [],
        "has_tables": False,
        "tables": [],
        "has_diagrams": False,
        "diagrams": [],
        "key_text": [],
        "summary": "This is an important summary"
    }
    text = vision_description_to_text(vision_data)
    assert "This is an important summary" in text


def test_extract_images_returns_list(tmp_path):
    """extract_images_from_pdf should return a list even for empty PDFs."""
    import fitz
    # Create a minimal empty PDF
    doc = fitz.open()
    doc.new_page()
    pdf_path = str(tmp_path / "empty.pdf")
    doc.save(pdf_path)
    doc.close()

    images = extract_images_from_pdf(pdf_path)
    assert isinstance(images, list)


def test_pdf_page_to_base64_returns_string(tmp_path):
    """pdf_page_to_base64 should return a non-empty string."""
    import fitz
    import base64

    # Create minimal PDF with text
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Test content for base64 conversion")
    pdf_path = str(tmp_path / "test.pdf")
    doc.save(pdf_path)
    doc.close()

    b64 = pdf_page_to_base64(pdf_path, 0)

    assert isinstance(b64, str)
    assert len(b64) > 0

    # Verify it is valid base64
    try:
        decoded = base64.b64decode(b64)
        assert len(decoded) > 0
    except Exception:
        pytest.fail("Result is not valid base64")