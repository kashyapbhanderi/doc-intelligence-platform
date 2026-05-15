import os
import sys
import json
sys.path.insert(0, os.path.abspath('.'))

from dotenv import load_dotenv
load_dotenv()

from ingestion.vision_extractor import (
    pdf_page_to_base64,
    describe_page_with_vision
)


def test_vision_on_pdf(pdf_path: str, page_num: int = 0):
    """
    Test vision extraction on one page of a PDF.
    Costs approximately $0.01 per page with GPT-4o.
    """
    print(f"Testing vision on: {pdf_path}")
    print(f"Page: {page_num + 1}")
    print("-" * 40)

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in .env file")
        return

    # Convert page to image
    print("Converting page to image...")
    b64 = pdf_page_to_base64(pdf_path, page_num)
    print(f"Image size: {len(b64)} characters (base64)")

    # Send to Vision API
    print("Sending to GPT-4o Vision...")
    result = describe_page_with_vision(b64, page_num)

    # Show results
    print("\nVision API Response:")
    print(json.dumps(result, indent=2))

    # Summary
    print("\nSummary:")
    print(f"  Has charts:   {result.get('has_charts', False)}")
    print(f"  Has tables:   {result.get('has_tables', False)}")
    print(f"  Has diagrams: {result.get('has_diagrams', False)}")
    print(f"  Summary: {result.get('summary', 'none')}")


if __name__ == "__main__":
    pdf = "data/raw/sample3.pdf"
    if os.path.exists(pdf):
        test_vision_on_pdf(pdf, page_num=0)
    else:
        print("PDF not found. Run: python scripts/download_samples.py")