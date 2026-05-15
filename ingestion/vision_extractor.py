import fitz  # PyMuPDF
import os
import base64
import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


def get_openai_client():
    """
    Create OpenAI client.
    Reads API key from .env file automatically.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in .env file. "
            "Add it like: OPENAI_API_KEY=sk-your-key"
        )
    return OpenAI(api_key=api_key)


def pdf_page_to_base64(pdf_path: str, page_num: int, dpi: int = 150) -> str:
    """
    Convert a single PDF page to a base64 encoded PNG image.

    Why base64? OpenAI Vision API accepts images as base64 strings.
    DPI 150 is a good balance — clear enough for OCR, not too large.

    Args:
        pdf_path: Path to the PDF file
        page_num: Page number (0-indexed)
        dpi: Image resolution (higher = clearer but slower)

    Returns:
        Base64 encoded string of the page image
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    # zoom factor: 150 DPI / 72 default DPI = 2.08x zoom
    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    # render page to pixel map
    pix = page.get_pixmap(matrix=mat)

    # convert to bytes then base64
    img_bytes = pix.tobytes("png")
    base64_str = base64.b64encode(img_bytes).decode("utf-8")

    doc.close()
    return base64_str


def describe_page_with_vision(base64_image: str, page_num: int) -> dict:
    """
    Send a page image to GPT-4o Vision and get structured description.

    The prompt asks specifically for:
    - Charts and their data
    - Tables and their content
    - Key visual information text would miss

    Args:
        base64_image: Base64 encoded page image
        page_num: Page number for reference

    Returns:
        Dict with description and extracted data
    """
    client = get_openai_client()

    prompt = """Analyze this document page carefully. Extract ALL information including:

1. CHARTS: Describe any charts/graphs. What type? What data does it show? 
   List approximate values if visible.

2. TABLES: Extract table content. List all rows and columns with their values.

3. IMAGES/DIAGRAMS: Describe any diagrams, flowcharts, or images.

4. KEY TEXT: Any important text that appears as an image (not regular text).

Return your response as JSON with this exact structure:
{
  "has_charts": true or false,
  "has_tables": true or false,
  "has_diagrams": true or false,
  "charts": ["description of chart 1", "description of chart 2"],
  "tables": ["table content as text"],
  "diagrams": ["description of diagram"],
  "key_text": ["any important text found in images"],
  "summary": "one sentence summary of visual content on this page"
}

If no visual elements found, return the structure with empty lists and false values."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}",
                                "detail": "high"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            max_tokens=1000
        )

        raw_response = response.choices[0].message.content

        # clean response — remove markdown code blocks if present
        clean = raw_response.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]

        result = json.loads(clean.strip())
        result["page"] = page_num + 1
        result["method"] = "gpt4o_vision"
        return result

    except json.JSONDecodeError:
        # if JSON parsing fails, return raw text
        return {
            "page": page_num + 1,
            "method": "gpt4o_vision",
            "has_charts": False,
            "has_tables": False,
            "has_diagrams": False,
            "charts": [],
            "tables": [],
            "diagrams": [],
            "key_text": [],
            "summary": raw_response[:500],
            "parse_error": True
        }

    except Exception as e:
        return {
            "page": page_num + 1,
            "method": "gpt4o_vision",
            "error": str(e),
            "has_charts": False,
            "has_tables": False,
            "has_diagrams": False,
            "charts": [],
            "tables": [],
            "diagrams": [],
            "key_text": [],
            "summary": ""
        }

def vision_description_to_text(vision_data: dict) -> str:
    """
    Convert vision API response dict into readable text.
    This text gets chunked and embedded alongside regular text.

    Args:
        vision_data: Dict returned by describe_page_with_vision()

    Returns:
        Plain text string of all visual content found
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

def extract_images_from_pdf(pdf_path: str) -> list:
    """
    Extract all embedded images from a PDF.

    This finds actual image files embedded in the PDF
    (logos, photos, charts saved as images).

    Returns list of dicts with image data and location.
    """
    doc = fitz.open(pdf_path)
    images = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]  # image reference number

            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_ext = base_image["ext"]

                # convert to base64
                b64 = base64.b64encode(img_bytes).decode("utf-8")

                images.append({
                    "page": page_num + 1,
                    "image_index": img_index,
                    "extension": img_ext,
                    "size_bytes": len(img_bytes),
                    "base64": b64,
                    "source": os.path.basename(pdf_path)
                })

            except Exception as e:
                print(f"  Could not extract image {img_index} "
                      f"on page {page_num + 1}: {e}")

    doc.close()
    print(f"Found {len(images)} embedded images in {os.path.basename(pdf_path)}")
    return images


if __name__ == "__main__":
    print("Vision extractor loaded successfully.")
    print("Functions available:")
    print("  - pdf_page_to_base64(pdf_path, page_num)")
    print("  - describe_page_with_vision(base64_image, page_num)")
    print("  - extract_images_from_pdf(pdf_path)")