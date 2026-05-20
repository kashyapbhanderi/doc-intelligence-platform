"""
agents/editor_tools/image_tools.py
Image processing tools using Pillow + rembg.

Pillow = industry standard Python image library
rembg  = AI-powered background removal
         (uses U2Net model under the hood)
"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter


# ── Image info ────────────────────────────────────────────

def get_image_info(path: str) -> dict:
    """Get metadata about an image file."""
    if not os.path.exists(path):
        return {"error": f"File not found: {path}"}

    img = Image.open(path)
    return {
        "filename": Path(path).name,
        "format":   img.format or Path(path).suffix,
        "mode":     img.mode,
        "size":     f"{img.width}×{img.height}px",
        "file_size": f"{os.path.getsize(path)//1024}KB",
    }


# ── Resize ────────────────────────────────────────────────

def resize_image(
    path: str,
    width: int = None,
    height: int = None,
    output_path: str = None
) -> str:
    """
    Resize an image while preserving aspect ratio.

    If only width OR height is given, the other
    dimension is calculated proportionally.
    This prevents distortion.

    Args:
        path:        Input image path
        width:       Target width in pixels
        height:      Target height in pixels
        output_path: Output path (default: adds _resized)

    Returns:
        Result message with new dimensions
    """
    if not os.path.exists(path):
        return f"Error: file not found: {path}"

    if not width and not height:
        return "Error: provide width or height"

    img = Image.open(path)
    orig_w, orig_h = img.size

    # Calculate missing dimension
    if width and not height:
        ratio  = width / orig_w
        height = int(orig_h * ratio)
    elif height and not width:
        ratio = height / orig_h
        width = int(orig_w * ratio)

    resized = img.resize(
        (width, height), Image.LANCZOS)

    if not output_path:
        stem = Path(path).stem
        ext  = Path(path).suffix
        output_path = str(
            Path(path).parent /
            f"{stem}_resized{ext}"
        )

    resized.save(output_path)
    return (f"Resized {orig_w}×{orig_h} → "
            f"{width}×{height}px → "
            f"{Path(output_path).name}")


# ── Watermark ─────────────────────────────────────────────

def add_text_watermark(
    path: str,
    text: str = "CONFIDENTIAL",
    position: str = "bottom-right",
    output_path: str = None
) -> str:
    """
    Add a text watermark to an image.

    Position options: top-left, top-right,
                      bottom-left, bottom-right,
                      center

    Args:
        path:        Input image path
        text:        Watermark text
        position:    Where to place the text
        output_path: Output path

    Returns:
        Result message
    """
    if not os.path.exists(path):
        return f"Error: file not found: {path}"

    img    = Image.open(path).convert("RGBA")
    draw   = ImageDraw.Draw(img)
    w, h   = img.size

    # Estimate text size (approx 20px font)
    font_size  = max(20, w // 25)
    text_w     = len(text) * font_size * 0.6
    text_h     = font_size

    padding = 20
    positions = {
        "top-left":     (padding, padding),
        "top-right":    (w - text_w - padding,
                         padding),
        "bottom-left":  (padding,
                         h - text_h - padding),
        "bottom-right": (w - text_w - padding,
                         h - text_h - padding),
        "center":       (w // 2 - text_w // 2,
                         h // 2 - text_h // 2),
    }
    pos = positions.get(position,
                        positions["bottom-right"])

    # Draw with semi-transparent background
    draw.text(
        pos, text,
        fill=(255, 0, 0, 180)   # red, 70% opacity
    )

    # Convert back to RGB for saving as JPEG/PNG
    output_img = img.convert("RGB")

    if not output_path:
        stem = Path(path).stem
        ext  = Path(path).suffix
        output_path = str(
            Path(path).parent /
            f"{stem}_watermarked{ext}"
        )

    output_img.save(output_path)
    return (f"Watermark '{text}' added at "
            f"{position} → "
            f"{Path(output_path).name}")


# ── Format conversion ─────────────────────────────────────

def convert_image_format(
    path: str,
    output_format: str = "png",
    output_path: str = None
) -> str:
    """
    Convert image between formats.
    Supports: png, jpg, jpeg, webp, bmp, tiff

    Args:
        path:          Input image path
        output_format: Target format (without dot)
        output_path:   Output path

    Returns:
        Result message
    """
    if not os.path.exists(path):
        return f"Error: file not found: {path}"

    fmt = output_format.lower().lstrip(".")
    supported = {"png", "jpg", "jpeg",
                 "webp", "bmp", "tiff"}

    if fmt not in supported:
        return (f"Error: unsupported format '{fmt}'. "
                f"Use: {supported}")

    img = Image.open(path)

    # JPEG doesn't support transparency
    if fmt in {"jpg", "jpeg"} and \
       img.mode in {"RGBA", "P"}:
        img = img.convert("RGB")

    if not output_path:
        output_path = str(
            Path(path).with_suffix(f".{fmt}"))

    img.save(output_path)
    size_kb = os.path.getsize(output_path) // 1024
    return (f"Converted to {fmt.upper()}: "
            f"{Path(output_path).name} "
            f"({size_kb}KB)")


# ── Background removal ────────────────────────────────────

def remove_background(
    path: str,
    output_path: str = None
) -> str:
    """
    Remove background from an image using AI.

    Uses rembg which runs U2Net — a neural network
    trained specifically for background segmentation.
    Works best on product photos and portraits.
    Output is always PNG (needs transparency).

    Args:
        path:        Input image path
        output_path: Output .png path

    Returns:
        Result message
    """
    if not os.path.exists(path):
        return f"Error: file not found: {path}"

    try:
        from rembg import remove

        img    = Image.open(path)
        result = remove(img)   # AI background removal

        if not output_path:
            output_path = str(
                Path(path).with_suffix(
                    "_no_bg.png"
                ).with_stem(
                    Path(path).stem + "_no_bg"
                )
            )

            # JPEG doesn't support transparency — auto-convert
            # to PNG or composite on white background
            ext = Path(output_path).suffix.lower()
            if ext in {".jpg", ".jpeg"}:
                # Place transparent image on white background
                white_bg = Image.new("RGB",
                                    result.size,
                                    (255, 255, 255))
                white_bg.paste(result,
                            mask=result.split()[3])  # alpha channel as mask
                white_bg.save(output_path)
            else:
                # PNG keeps transparency — ideal for bg removal
                result.save(output_path)

            return (f"Background removed → "
                    f"{Path(output_path).name}")

    except ImportError:
        return ("rembg not installed. "
                    "Run: pip install rembg")
    except Exception as e:
        return f"Error: {e}"


if __name__ == "__main__":
    TEST_IMG = "data/test_docs/test_image.png"

    if not os.path.exists(TEST_IMG):
        print("Test image missing. Run:")
        print("  python scripts/create_test_docs.py")
    else:
        print("Testing Image Tools")
        print("=" * 50)

        # Info
        info = get_image_info(TEST_IMG)
        print(f"Image info: {info['size']} "
              f"{info['format']}")

        # Resize
        result = resize_image(
            TEST_IMG, width=400,
            output_path="data/test_docs/"
                        "test_image_400w.png"
        )
        print(f"Resize:     {result}")

        # Watermark
        result = add_text_watermark(
            TEST_IMG, "SAMPLE",
            output_path="data/test_docs/"
                        "test_image_wm.png"
        )
        print(f"Watermark:  {result}")

        # Convert format
        result = convert_image_format(
            TEST_IMG, "jpg",
            output_path="data/test_docs/"
                        "test_image.jpg"
        )
        print(f"Convert:    {result}")

        # bd remove
        result=remove_background(
            TEST_IMG, 
            output_path="data/test_docs/"
                        "test_image.png"
        )
        print(f"remove:    {result}")

        print("\n✅ All image tools working!")
        