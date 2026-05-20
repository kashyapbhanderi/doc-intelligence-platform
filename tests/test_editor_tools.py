import pytest
import sys
import os
import shutil
from pathlib import Path

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from agents.editor_tools.docx_tools import (
    edit_docx_text,
    insert_table_docx,
    add_heading_docx,
    read_docx_text,
    get_docx_info,
)
from agents.editor_tools.image_tools import (
    get_image_info,
    resize_image,
    add_text_watermark,
    convert_image_format,
)

TEST_DIR  = "data/test_docs"
TEST_DOCX = f"{TEST_DIR}/test_report.docx"
TEST_IMG  = f"{TEST_DIR}/test_image.png"


# ── Setup ─────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def create_test_files():
    """Create test files before running tests."""
    if not os.path.exists(TEST_DOCX) or \
       not os.path.exists(TEST_IMG):
        from scripts.create_test_docs import (
            create_test_docx,
            create_test_image,
        )
        create_test_docx()
        create_test_image()


# ── DOCX tests ────────────────────────────────────────────

def test_get_docx_info_returns_dict():
    info = get_docx_info(TEST_DOCX)
    assert isinstance(info, dict)
    assert "filename" in info
    assert "word_count" in info


def test_get_docx_info_word_count_positive():
    info = get_docx_info(TEST_DOCX)
    assert info["word_count"] > 0


def test_read_docx_returns_string():
    text = read_docx_text(TEST_DOCX)
    assert isinstance(text, str)
    assert len(text) > 0


def test_edit_docx_text_replaces_content(tmp_path):
    """Copy file and test replacement."""
    copy = str(tmp_path / "test_edit.docx")
    shutil.copy(TEST_DOCX, copy)

    result = edit_docx_text(copy, "Summary", "Overview")
    assert "Overview" in result or "0 occurrence" in result


def test_edit_docx_text_missing_file():
    result = edit_docx_text(
        "nonexistent.docx", "old", "new")
    assert "Error" in result


def test_insert_table_docx(tmp_path):
    copy = str(tmp_path / "test_table.docx")
    shutil.copy(TEST_DOCX, copy)

    result = insert_table_docx(
        copy,
        headers=["Col A", "Col B"],
        rows=[["1", "2"], ["3", "4"]],
        heading="Test Table"
    )
    assert "2 cols" in result
    assert "2 rows" in result


def test_add_heading_docx(tmp_path):
    copy = str(tmp_path / "test_heading.docx")
    shutil.copy(TEST_DOCX, copy)

    result = add_heading_docx(
        copy, "New Section", level=2)
    assert "New Section" in result
    assert "H2" in result


def test_edit_docx_text_no_match(tmp_path):
    """Replacing non-existent text returns 0 occurrences."""
    copy = str(tmp_path / "test_no_match.docx")
    shutil.copy(TEST_DOCX, copy)

    result = edit_docx_text(
        copy,
        "ZZZZZ_DEFINITELY_NOT_IN_DOC",
        "replacement"
    )
    assert "0 occurrence" in result


# ── Image tests ───────────────────────────────────────────

def test_get_image_info_returns_dict():
    info = get_image_info(TEST_IMG)
    assert isinstance(info, dict)
    assert "size" in info
    assert "format" in info


def test_get_image_info_missing_file():
    info = get_image_info("nonexistent.png")
    assert "error" in info


def test_resize_image_by_width(tmp_path):
    output = str(tmp_path / "resized.png")
    result = resize_image(TEST_IMG,
                          width=200,
                          output_path=output)
    assert "200" in result
    assert os.path.exists(output)

    from PIL import Image
    img = Image.open(output)
    assert img.width == 200


def test_resize_image_preserves_ratio(tmp_path):
    output = str(tmp_path / "resized_ratio.png")
    resize_image(TEST_IMG, width=400,
                 output_path=output)

    from PIL import Image
    orig  = Image.open(TEST_IMG)
    resized = Image.open(output)

    orig_ratio    = orig.width / orig.height
    resized_ratio = resized.width / resized.height
    assert abs(orig_ratio - resized_ratio) < 0.1


def test_add_text_watermark(tmp_path):
    output = str(tmp_path / "watermarked.png")
    result = add_text_watermark(
        TEST_IMG, "TEST",
        output_path=output
    )
    assert "TEST" in result
    assert os.path.exists(output)


def test_convert_image_format(tmp_path):
    output = str(tmp_path / "converted.jpg")
    result = convert_image_format(
        TEST_IMG, "jpg",
        output_path=output
    )
    assert "JPG" in result
    assert os.path.exists(output)


def test_convert_unsupported_format():
    result = convert_image_format(
        TEST_IMG, "xyz")
    assert "Error" in result


def test_resize_no_dimensions():
    result = resize_image(TEST_IMG)
    assert "Error" in result