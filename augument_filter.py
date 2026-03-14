"""
Augmentation script: 1 image -> 7 filters x 3 augmented variants = 21 outputs.
Same purpose and augmentation style as OCR_Model/agument_obb.py (SafeRotate, Affine,
RandomBrightnessContrast, Blur with keypoint-aware transform), then apply the 7 OCR
filters (0,1,3,5,border,8,9) to each augmented image. No mirror so text stays readable.
"""
from pathlib import Path
import sys
import cv2
import numpy as np

# Project root so we can import from ipv
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ipv import _ocr_apply_try_filter

try:
    import albumentations as A
except ImportError:
    raise SystemExit("Install albumentations: pip install albumentations")

# 7 filters (same order as ipv): 0, 1, 3, 5, border, 8, 9
NUM_FILTERS = 7
# 3 augmented variants per image (same as agument_obb.py)
NUM_AUG_VARIANTS = 3

INPUT_DIR = PROJECT_ROOT / "OCR_Model" / "train"
OUTPUT_DIR = PROJECT_ROOT / "OCR_Model" / "train_augmented_filter"


def build_transform():
    """Same as OCR_Model/agument_obb.py: SafeRotate, Affine, RandomBrightnessContrast, Blur."""
    return A.Compose([
        A.SafeRotate(limit=20, p=0.8, border_mode=cv2.BORDER_CONSTANT, fill_value=0),
        A.Affine(
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
            scale=(0.9, 1.1),
            rotate=0,
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.Blur(blur_limit=3, p=0.2),
    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def parse_obb_label(label_path):
    """
    Reads a YOLO OBB label file.
    Each line: class x1 y1 x2 y2 x3 y3 x4 y4  (all normalized 0-1)
    Returns: (class_labels, polygons) where polygons is list of (4, 2) arrays.
    """
    class_labels = []
    polygons = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            class_labels.append(int(float(parts[0])))
            coords = [float(x) for x in parts[1:9]]
            poly = np.array(coords, dtype=np.float64).reshape(4, 2)
            polygons.append(poly)
    return class_labels, polygons


def polygons_to_keypoints(polygons, img_w, img_h):
    """Flatten polygon corners to keypoints in absolute pixel coords (same as agument_obb.py)."""
    keypoints = []
    for poly in polygons:
        for (nx, ny) in poly:
            keypoints.append((nx * img_w, ny * img_h))
    return keypoints


def keypoints_to_polygons(keypoints, img_w, img_h, n_polygons):
    """Rebuild normalized polygons from transformed keypoints (same as agument_obb.py)."""
    polygons = []
    for i in range(n_polygons):
        pts = keypoints[i * 4 : i * 4 + 4]
        poly = []
        for (px, py) in pts:
            nx = np.clip(px / img_w, 0.0, 1.0)
            ny = np.clip(py / img_h, 0.0, 1.0)
            poly.append((nx, ny))
        polygons.append(poly)
    return polygons


def is_valid_polygon(poly):
    """Reject collapsed polygons (same as agument_obb.py)."""
    pts = np.array(poly, dtype=np.float32)
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area > 1e-4


def save_obb_label(label_path, class_labels, polygons):
    with open(label_path, "w") as f:
        for cls, poly in zip(class_labels, polygons):
            coords = " ".join(f"{v:.6f}" for pt in poly for v in pt)
            f.write(f"{cls} {coords}\n")


def augment_with_filters():
    transform = build_transform()
    input_images = INPUT_DIR / "images"
    input_labels = INPUT_DIR / "labels"
    output_images = OUTPUT_DIR / "images"
    output_labels = OUTPUT_DIR / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = []
    for ext in exts:
        image_files.extend(input_images.glob(f"*{ext}"))
    image_files = sorted(set(image_files))

    if not image_files:
        print(f"No images found in {input_images}")
        return

    print(f"Input: {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Augmentation: same as agument_obb.py (SafeRotate, Affine, BrightnessContrast, Blur)")
    print(f"Then {NUM_FILTERS} filters x {NUM_AUG_VARIANTS} variants = {NUM_FILTERS * NUM_AUG_VARIANTS} per image")
    print(f"Processing {len(image_files)} images...")

    total_saved = 0
    for img_path in image_files:
        label_path = input_labels / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"  Skip (no label): {img_path.name}")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Skip (load failed): {img_path.name}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]
        class_labels, polygons = parse_obb_label(label_path)
        if not polygons:
            print(f"  Skip (no polygons): {img_path.name}")
            continue

        base_name = img_path.stem
        count_this = 0
        for i in range(NUM_AUG_VARIANTS):
            try:
                keypoints = polygons_to_keypoints(polygons, img_w, img_h)
                result = transform(image=img_rgb, keypoints=keypoints)
                aug_img_rgb = result["image"]
                aug_h, aug_w = aug_img_rgb.shape[:2]
                aug_polygons = keypoints_to_polygons(
                    result["keypoints"], aug_w, aug_h, len(polygons)
                )
                valid = [
                    (cls, poly)
                    for cls, poly in zip(class_labels, aug_polygons)
                    if is_valid_polygon(poly)
                ]
                if not valid:
                    continue
                aug_labels, aug_polys = zip(*valid)
                aug_labels, aug_polys = list(aug_labels), list(aug_polys)
                aug_img_bgr = cv2.cvtColor(aug_img_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"  Skip aug {i} for {img_path.name}: {e}")
                continue

            for filter_idx in range(NUM_FILTERS):
                filtered = _ocr_apply_try_filter(aug_img_bgr, filter_idx)
                if filtered is None:
                    continue
                out_name = f"{base_name}_aug_{i}_f{filter_idx}"
                out_img_path = output_images / f"{out_name}.png"
                out_label_path = output_labels / f"{out_name}.txt"
                cv2.imwrite(str(out_img_path), filtered)
                save_obb_label(out_label_path, aug_labels, aug_polys)
                count_this += 1
        total_saved += count_this
        print(f"  {img_path.name} -> {count_this} variants")

    print(f"\nDone. Total saved: {total_saved} images (+ labels) in {OUTPUT_DIR}")


if __name__ == "__main__":
    augment_with_filters()
