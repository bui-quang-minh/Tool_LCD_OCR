"""
Run OCR_Model (YOLO OBB) on chart/sample.png and save annotated result.
Useful to demonstrate OCR / SVM effectiveness on a single image.

Run from project root: python chart/run_ocr_sample.py
Requirements: same as ipv.py (ultralytics, opencv-python, numpy)
"""
import time
from pathlib import Path
import sys

# Allow importing from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from ipv import (
    load_ocr_model,
    _ocr_format_with_retries,
    _ocr_validate_format,
    OCR_MODEL_WEIGHTS,
    OCR_PADDING,
    OCR_FILTER_NAMES,
)

CHART_DIR = Path(__file__).resolve().parent
SAMPLE_IMAGE = CHART_DIR / "sample.png"
OUTPUT_ANNOTATED = CHART_DIR / "ocr_sample_result.png"


def main():
    if not SAMPLE_IMAGE.exists():
        print(f"Image not found: {SAMPLE_IMAGE}")
        return 1

    img = cv2.imread(str(SAMPLE_IMAGE))
    if img is None:
        print(f"Failed to load image: {SAMPLE_IMAGE}")
        return 1

    if not OCR_MODEL_WEIGHTS.exists():
        print(f"OCR model not found: {OCR_MODEL_WEIGHTS}")
        return 1

    print("Loading OCR model...")
    ocr_model = load_ocr_model()
    if ocr_model is None:
        print("Failed to load OCR model.")
        return 1
    # Same padding as in ipv so edge characters are not cut off
    h, w = img.shape[:2]
    padded = cv2.copyMakeBorder(
        img, OCR_PADDING, OCR_PADDING, OCR_PADDING, OCR_PADDING,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )

    print("Running OCR with filter retries...")
    start_time = time.time()

    display_str, vis, n_tries, filter_idx, raw_str = _ocr_format_with_retries(ocr_model, padded)
    end_time = time.time() # Kết thúc đo
    latency = (end_time - start_time) * 1000
    print(f"--- Inference Latency: {latency:.2f} ms ---")
    # vis is on padded image; crop back to original size for output if desired, or keep full
    # Here we keep the padded annotated image so boxes are visible
    cv2.imwrite(str(OUTPUT_ANNOTATED), vis)
    print(f"Saved annotated image: {OUTPUT_ANNOTATED}")

    filter_name = OCR_FILTER_NAMES[filter_idx] if 0 <= filter_idx < len(OCR_FILTER_NAMES) else f"filter_{filter_idx}"
    print(f"Recognized (raw): {raw_str}")
    print(f"Recognized (display): {display_str}")
    print(f"Filter used: {filter_name} (attempt {n_tries})")

    # OK/NG verdict for xx.x nm (60–80 Nm)
    if display_str:
        clean = display_str.split(" (")[0].strip().lower()
        if _ocr_validate_format(clean):
            try:
                val = float(clean.replace("nm", "").replace("mp", "").strip())
                if 60 <= val <= 80:
                    print("Verdict: OK (60–80 Nm)")
                else:
                    print("Verdict: NG (outside 60–80 Nm)")
            except ValueError:
                print("Verdict: — (unparseable)")
        else:
            print("Verdict: — (format not xx.xnm)")
    else:
        print("Verdict: — (no recognition)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
