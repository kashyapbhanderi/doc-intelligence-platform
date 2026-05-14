import urllib.request
import os

os.makedirs("data/raw", exist_ok=True)

# Free public domain PDFs for testing
samples = [
    (
        "https://www.w3.org/WAI/WCAG21/Techniques/pdf/pdf-sample.pdf",
        "data/raw/sample1.pdf"
    ),
    (
        "https://www.africau.edu/images/default/sample.pdf",
        "data/raw/sample2.pdf"
    ),
    (
        "https://www.orimi.com/pdf-test.pdf",
        "data/raw/sample3.pdf"
    ),
]

for url, path in samples:
    print(f"Downloading {path}...")
    try:
        urllib.request.urlretrieve(url, path)
        size = os.path.getsize(path)
        print(f"  Saved: {size} bytes")
    except Exception as e:
        print(f"  Failed: {e}")

print("\nDone! Check data/raw/ folder.")