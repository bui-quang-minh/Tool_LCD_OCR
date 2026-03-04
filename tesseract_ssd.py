"""
Tesseract OCR using the 'ssd' (seven-segment digit) language.
Configured for display format: 2 digits, 1 decimal point, 1 digit after (e.g. 82.1).
"""
import re

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def read_7segment_tesseract(image_bgr_or_gray, digits_before=2, digits_after=1):
    """
    Run Tesseract with lang=ssd (seven-segment) and optional format constraint.
    Args:
        image_bgr_or_gray: numpy image (BGR or grayscale).
        digits_before: number of digits before decimal (default 2).
        digits_after: number of digits after decimal (default 1).
    Returns:
        Recognized text, optionally formatted as "XX.X" (e.g. "82.1").
    """
    try:
        import pytesseract
        import cv2
    except ImportError:
        return "(install pytesseract and opencv-python)"

    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    if image_bgr_or_gray is None or image_bgr_or_gray.size == 0:
        return "(no image)"
    if len(image_bgr_or_gray.shape) == 3:
        img = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2RGB)
    else:
        img = image_bgr_or_gray

    # Use seven-segment language 'ssd', single line, digits + decimal only
    config = (
        f"--psm 7 "  # single text line
        f"-c tessedit_char_whitelist=0123456789. "
        f"-c tessedit_char_blacklist= "
    )
    try:
        text = pytesseract.image_to_string(img, lang="ssd", config=config.strip())
    except Exception as e:
        return f"Tesseract error: {e}"

    text = (text or "").strip().replace(" ", "")
    if not text:
        return "(no text)"

    # Format to "X.X" or "XX.X": at most digits_before before decimal, digits_after after
    formatted = _format_reading(text, digits_before=digits_before, digits_after=digits_after)
    return formatted


def _format_reading(raw, digits_before=2, digits_after=1):
    """
    Extract a reading in form XX.X from raw string (digits and one decimal).
    - digits_before: max digits before decimal (e.g. 2 -> 82 or 8).
    - digits_after: digits after decimal (e.g. 1 -> .1).
    """
    # Keep only digits and first decimal
    allowed = re.sub(r"[^0-9.]", "", raw)
    parts = allowed.split(".", 1)
    before = re.sub(r"[^0-9]", "", parts[0])[: digits_before]
    after = (re.sub(r"[^0-9]", "", parts[1])[: digits_after]) if len(parts) > 1 else ""
    if not before and not after:
        return raw.strip() or "(no text)"
    if after:
        return f"{before}.{after}"
    return before


if __name__ == "__main__":
    import sys
    import cv2
    if len(sys.argv) < 2:
        print("Usage: python tesseract_ssd.py <image_path> [digits_before=2] [digits_after=1]")
        sys.exit(1)
    path = sys.argv[1]
    digits_before = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    digits_after = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    img = cv2.imread(path)
    if img is None:
        print("Could not load image:", path)
        sys.exit(2)
    result = read_7segment_tesseract(img, digits_before=digits_before, digits_after=digits_after)
    print(result)
