"""
Show how the 7 OCR retry filters (0,1,3,5,border,8,9) look on a sample image from test_image folder.
Run from project root: python test.py
"""
from pathlib import Path
import cv2
import numpy as np
import sys

# Add project root so we can import from ipv
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ipv import _ocr_apply_try_filter

# Prefer OCR_Model/test_image, then test_image at project root
for _dir in (PROJECT_ROOT / "OCR_Model" / "test_image", PROJECT_ROOT / "test_image"):
    if _dir.exists():
        TEST_IMAGE_DIR = _dir
        break
else:
    TEST_IMAGE_DIR = PROJECT_ROOT / "OCR_Model" / "test_image"

def _find_sample_image():
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    for ext in exts:
        for p in TEST_IMAGE_DIR.glob(f"*{ext}"):
            return p
    return None

SAMPLE_IMAGE = TEST_IMAGE_DIR / "5f5a1335-23_lcd0.png" if (TEST_IMAGE_DIR / "5f5a1335-23_lcd0.png").exists() else _find_sample_image()

FILTER_NAMES = [
    "0: Original",
    "1: Grayscale",
    "3: Binary (adaptive)",
    "5: Grayscale + Resize 2x",
    "Border: Morph gradient (kernel)",
    "8: Invert",
    "9: Contrast",
]
NUM_FILTERS = 7


def main():
    sample = SAMPLE_IMAGE if SAMPLE_IMAGE is not None else _find_sample_image()
    if sample is None or not sample.exists():
        print(f"No sample image found in: {TEST_IMAGE_DIR}")
        print("Add an image (e.g. .png, .jpg) there and run again.")
        return
    img = cv2.imread(str(sample))
    if img is None:
        print(f"Could not load image: {sample}")
        return
    print(f"Using: {sample}")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # Fallback: show each in an OpenCV window (press key to advance)
        print("Install matplotlib for grid view: pip install matplotlib")
        print("Showing each filter in OpenCV window (press any key for next)...")
        for i in range(NUM_FILTERS):
            out = _ocr_apply_try_filter(img, i)
            if out is not None:
                disp = cv2.resize(out, (400, 300)) if max(out.shape[:2]) > 500 else out
                cv2.imshow(FILTER_NAMES[i], disp)
                cv2.waitKey(0)
                cv2.destroyWindow(FILTER_NAMES[i])
        cv2.destroyAllWindows()
        return

    # Grid 2 rows x 4 columns (7 filters: 0,1,3,5,border,8,9)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle("OCR retry filters (0,1,3,5,border,8,9) — sample from test_image", fontsize=12)
    for i in range(NUM_FILTERS):
        row, col = i // 4, i % 4
        ax = axes[row, col]
        out = _ocr_apply_try_filter(img, i)
        if out is not None:
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            ax.imshow(rgb)
        ax.set_title(FILTER_NAMES[i], fontsize=9)
        ax.axis("off")
    axes[1, 3].axis("off")  # hide empty 8th cell
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
