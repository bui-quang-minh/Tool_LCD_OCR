"""
Run the same 7 OCR retry filters as ipv._ocr_apply_try_filter on every image
in OCR_Model/test_image (or ./test_image), save outputs under test_image/result/,
and write FILTERS_README.txt describing each filter.

For Binary (adaptive), also saves grayscale before vs after Gaussian denoise.

Run from project root: python export_test_image_filters.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ipv import _ocr_apply_try_filter

for _dir in (PROJECT_ROOT / "OCR_Model" / "test_image", PROJECT_ROOT / "test_image"):
    if _dir.exists():
        TEST_IMAGE_DIR = _dir
        break
else:
    TEST_IMAGE_DIR = PROJECT_ROOT / "OCR_Model" / "test_image"

RESULT_DIR = TEST_IMAGE_DIR / "result"
NUM_FILTERS = 7

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}

FILTER_FILE_PREFIXES = [
    "00_original",
    "01_grayscale",
    "02_binary_adaptive",
    "03_grayscale_resize_2x",
    "04_border_morph_gradient",
    "05_invert",
    "06_contrast",
]


def _binary_adaptive_denoise_steps(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match ipv._ocr_bgr_to_binary_adaptive: gray, denoised gray, final BGR binary."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(g, (3, 3), 0)
    bin_gray = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5
    )
    out_bgr = cv2.cvtColor(bin_gray, cv2.COLOR_GRAY2BGR)
    return g, denoised, out_bgr


def _readme_text() -> str:
    return """Seven OCR retry filters (same order as ipv._ocr_apply_try_filter / OCR_FILTER_NAMES)
================================================================================

These are applied one per retry when parsing LCD readouts; indices 0..6 map to
legacy try indices (0, 1, 3, 5, border, 8, 9).

1) Original (try 0)
   No change — input BGR crop is passed through as-is.

2) Grayscale (try 1)
   BGR -> single-channel gray, then gray -> 3-channel BGR (still gray pixels)
   so the OCR model always receives a 3-channel image.

3) Binary (adaptive) (try 3)
   - Convert to grayscale.
   - Denoise: Gaussian blur kernel (3, 3), sigma 0 (OpenCV default).
   - Binarize: adaptive Gaussian threshold, block size 11, constant subtracted
     from mean C = 3.5, output binary 0/255.
   This script also writes *_denoise_before.png (raw grayscale) and
   *_denoise_after.png (after Gaussian blur, before threshold) so you can see
   the denoise step alone.

4) Grayscale + resize 2x (try 5)
   BGR -> gray, then bicubic upscale (fx=2, fy=2), then gray -> BGR. Helps when
   characters are small relative to image size.

5) Border — morphological gradient (kernel) (border try)
   Gray -> morphological gradient (dilate minus erode) with a 3x3 rectangular
   kernel, then BGR. Emphasizes edges/outlines of segments.

6) Invert (try 8)
   Gray -> bitwise NOT -> BGR. Useful when text/background polarity is wrong
   for the model.

7) Contrast (try 9)
   Gray -> cv2.normalize to stretch intensity to full 0..255 (min-max), then BGR.
   Improves low-contrast crops.

References: ipv.py — _ocr_apply_try_filter, _ocr_bgr_to_binary_adaptive,
_ocr_bgr_to_border, _ocr_invert, _ocr_contrast.
"""


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    readme_path = RESULT_DIR / "FILTERS_README.txt"
    readme_path.write_text(_readme_text(), encoding="utf-8")

    images = sorted(
        p
        for p in TEST_IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    # Exclude anything already under result/ by only listing direct children; skip if name is 'result' folder files are not in iterdir of test_image for subfolders - actually iterdir only top level. Good.

    if not images:
        print(f"No images found in {TEST_IMAGE_DIR}")
        return

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Skip (unreadable): {img_path}")
            continue

        base = img_path.stem
        out_dir = RESULT_DIR / base
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(NUM_FILTERS):
            prefix = FILTER_FILE_PREFIXES[i]
            filtered = _ocr_apply_try_filter(img, i)
            if filtered is None:
                continue
            cv2.imwrite(str(out_dir / f"{prefix}.png"), filtered)

            if i == 2:
                g_before, g_after, _ = _binary_adaptive_denoise_steps(img)
                cv2.imwrite(str(out_dir / f"{prefix}_denoise_before.png"), g_before)
                cv2.imwrite(str(out_dir / f"{prefix}_denoise_after.png"), g_after)

        print(f"Wrote: {out_dir}")

    print(f"Readme: {readme_path}")


if __name__ == "__main__":
    main()
