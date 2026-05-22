"""
agents/editor_tools/tool_registry.py

Wraps all editor functions as LangGraph @tool
decorated functions so the Editor Agent LLM
can call them by name.

How @tool works:
- LLM reads the function docstring to understand
  what the tool does
- LLM decides which tool to call based on the
  user instruction
- LLM fills in the arguments automatically
- Tool runs and returns a string result

Industry term: "Tool calling" or "Function calling"
Used by OpenAI, Anthropic, Google in production.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from langchain_core.tools import tool

from agents.editor_tools.docx_tools import (
    edit_docx_text,
    insert_table_docx,
    add_heading_docx,
    read_docx_text,
    get_docx_info,
)
from agents.editor_tools.pdf_tools import (
    get_pdf_info,
    watermark_pdf,
    merge_pdfs,
    split_pdf,
    pdf_to_docx,
)
from agents.editor_tools.image_tools import (
    get_image_info,
    resize_image,
    add_text_watermark,
    convert_image_format,
)


# ── DOCX tools ────────────────────────────────────────────

@tool
def tool_read_docx(file_path: str) -> str:
    """
    Read and return all text content from a Word
    document (.docx file). Use this FIRST before
    making any edits to understand the document.
    Args: file_path - path to the .docx file.
    """
    return read_docx_text(file_path)


@tool
def tool_edit_docx_text(
    file_path: str,
    old_text: str,
    new_text: str
) -> str:
    """
    Find and replace text in a Word document.
    Preserves original formatting.
    Use when the user wants to change specific
    words, phrases, dates, names, or terms.
    Args: file_path, old_text to find,
    new_text to replace with.
    """
    return edit_docx_text(file_path, old_text, new_text)


@tool
def tool_insert_table_docx(
    file_path: str,
    headers: list,
    rows: list,
    heading: str = ""
) -> str:
    """
    Insert a formatted table into a Word document.
    Appends the table at the end of the document.
    Use when the user wants to add structured data,
    comparisons, or lists in table format.
    Args: file_path, headers (list of column names),
    rows (list of lists), heading (optional title).
    """
    return insert_table_docx(
        file_path, headers, rows, heading)


@tool
def tool_add_heading_docx(
    file_path: str,
    heading_text: str,
    level: int = 2
) -> str:
    """
    Add a heading to a Word document.
    Level 1 is largest (H1), level 3 is smallest.
    Use when user wants to add a new section title.
    Args: file_path, heading_text, level (1-3).
    """
    return add_heading_docx(
        file_path, heading_text, level)


@tool
def tool_get_docx_info(file_path: str) -> str:
    """
    Get metadata about a Word document: word count,
    paragraph count, table count, and text preview.
    Use to understand document structure before editing.
    Args: file_path - path to the .docx file.
    """
    info = get_docx_info(file_path)
    return str(info)


# ── PDF tools ─────────────────────────────────────────────

@tool
def tool_watermark_pdf(
    file_path: str,
    watermark_text: str = "CONFIDENTIAL"
) -> str:
    """
    Add a diagonal text watermark to every page
    of a PDF file. Common watermarks: CONFIDENTIAL,
    DRAFT, COPY, SAMPLE.
    Args: file_path, watermark_text.
    """
    return watermark_pdf(file_path, watermark_text)


@tool
def tool_merge_pdfs(
    file_paths: list,
    output_path: str = "data/test_docs/merged.pdf"
) -> str:
    """
    Merge multiple PDF files into one document.
    Pages are appended in the order given.
    Use when user wants to combine PDF files.
    Args: file_paths (list of paths),
    output_path where to save the merged file.
    """
    return merge_pdfs(file_paths, output_path)


@tool
def tool_split_pdf(
    file_path: str,
    start_page: int = 1,
    end_page: int = 3
) -> str:
    """
    Extract a range of pages from a PDF into
    a new PDF file. Pages are 1-indexed.
    Use when user wants specific pages from a PDF.
    Args: file_path, start_page, end_page.
    """
    return split_pdf(file_path, start_page, end_page)


@tool
def tool_pdf_to_docx(
    file_path: str,
    output_path: str = None
) -> str:
    """
    Convert a PDF file to a Word document (.docx).
    Use when user wants to edit a PDF's content
    (edit as Word, save back if needed).
    Args: file_path to PDF, optional output_path.
    """
    return pdf_to_docx(file_path, output_path)


# ── Image tools ───────────────────────────────────────────

@tool
def tool_resize_image(
    file_path: str,
    width: int = 800
) -> str:
    """
    Resize an image to the specified width in pixels.
    Height is calculated automatically to preserve
    the original aspect ratio (no distortion).
    Args: file_path, width in pixels.
    """
    return resize_image(file_path, width=width)


@tool
def tool_watermark_image(
    file_path: str,
    text: str = "SAMPLE",
    position: str = "bottom-right"
) -> str:
    """
    Add a text watermark to an image.
    Position: top-left, top-right, bottom-left,
    bottom-right, or center.
    Args: file_path, text, position.
    """
    return add_text_watermark(
        file_path, text, position)


@tool
def tool_convert_image(
    file_path: str,
    output_format: str = "jpg"
) -> str:
    """
    Convert an image to a different format.
    Supported: png, jpg, jpeg, webp, bmp, tiff.
    Args: file_path, output_format (without dot).
    """
    return convert_image_format(
        file_path, output_format)


# ── Tool registry ─────────────────────────────────────────

ALL_EDITOR_TOOLS = [
    tool_read_docx,
    tool_edit_docx_text,
    tool_insert_table_docx,
    tool_add_heading_docx,
    tool_get_docx_info,
    tool_watermark_pdf,
    tool_merge_pdfs,
    tool_split_pdf,
    tool_pdf_to_docx,
    tool_resize_image,
    tool_watermark_image,
    tool_convert_image,
]

TOOL_NAMES = [t.name for t in ALL_EDITOR_TOOLS]


if __name__ == "__main__":
    print(f"Editor tools registered: "
          f"{len(ALL_EDITOR_TOOLS)}")
    print()
    for t in ALL_EDITOR_TOOLS:
        print(f"  {t.name}")
        desc = t.description.split("\n")[0]
        print(f"    → {desc[:70]}")