"""
t-SNE and UMAP visualization on a single LCD image.
Uses YOLO OBB annotations as ground-truth labels.
"""
from pathlib import Path
import numpy as np
import cv2

CHART_DIR = Path(__file__).resolve().parent
IMAGES_DIR = Path(r"F:\clone\Tool_LCD_OCR\OCR_Model\train\images")
LABELS_DIR = Path(r"F:\clone\Tool_LCD_OCR\OCR_Model\train\labels")
PATCH_SIZE = 16
OUTPUT_TSNE = CHART_DIR / "tsne.png"
OUTPUT_UMAP = CHART_DIR / "umap.png"


def parse_yolo_obb_label(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls_id = int(parts[0])
            coords = list(map(float, parts[1:9]))
            polygon = np.array([
                [coords[i] * img_w, coords[i + 1] * img_h]
                for i in range(0, 8, 2)
            ], dtype=np.int32)
            boxes.append((cls_id, polygon))
    return boxes


def patch_overlap_ratio(px, py, patch_size, polygons):
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    shifted = [poly - np.array([px, py]) for _, poly in polygons]
    for poly in shifted:
        cv2.fillPoly(mask, [poly], 255)
    return np.count_nonzero(mask) / (patch_size * patch_size)


def main():
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpg"))
    if not image_paths:
        print(f"Error: No images found in {IMAGES_DIR}")
        return

    img_path = image_paths[0]
    label_path = LABELS_DIR / (img_path.stem + ".txt")

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"Error: Cannot read {img_path}")
        return

    h, w = gray.shape[:2]
    polygons = parse_yolo_obb_label(label_path, w, h)
    print(f"Image: {img_path.name} ({w}x{h}), {len(polygons)} OBB labels")

    all_patches = []
    all_labels = []

    h2 = (h // PATCH_SIZE) * PATCH_SIZE
    w2 = (w // PATCH_SIZE) * PATCH_SIZE

    for y in range(0, h2, PATCH_SIZE):
        for x in range(0, w2, PATCH_SIZE):
            overlap = patch_overlap_ratio(x, y, PATCH_SIZE, polygons)
            if 0.10 <= overlap <= 0.70:
                continue
            label = 1 if overlap > 0.70 else 0
            patch = gray[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            all_patches.append(patch.ravel().astype(np.float32) / 255.0)
            all_labels.append(label)

    X = np.array(all_patches)
    y_labels = np.array(all_labels)
    print(f"Patches: {len(X)} (Background: {np.sum(y_labels==0)}, Digit: {np.sum(y_labels==1)})")

    if X.shape[0] < 2:
        print("Not enough patches.")
        return

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # t-SNE
    print("Running t-SNE...")
    perp = min(30, X.shape[0] - 1)
    X_tsne = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(X)

    plt.figure(figsize=(10, 7))
    plt.scatter(X_tsne[y_labels==0, 0], X_tsne[y_labels==0, 1],
                c='#D3D3D3', label='Background (0)', alpha=0.5, s=40,
                edgecolors='k', linewidth=0.3)
    plt.scatter(X_tsne[y_labels==1, 0], X_tsne[y_labels==1, 1],
                c='#8B0000', label='Digit Segment (1)', alpha=0.7, s=40,
                edgecolors='k', linewidth=0.3)
    plt.title("t-SNE: Data Separability Evaluation (Patch-based)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(OUTPUT_TSNE, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUTPUT_TSNE}")

    # UMAP
    try:
        import umap
        print("Running UMAP...")
        n_nb = min(15, X.shape[0] - 1)
        X_umap = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_nb).fit_transform(X)

        plt.figure(figsize=(10, 7))
        plt.scatter(X_umap[y_labels==0, 0], X_umap[y_labels==0, 1],
                    c='#D3D3D3', label='Background (0)', alpha=0.5, s=40,
                    edgecolors='k', linewidth=0.3)
        plt.scatter(X_umap[y_labels==1, 0], X_umap[y_labels==1, 1],
                    c='#8B0000', label='Digit Segment (1)', alpha=0.7, s=40,
                    edgecolors='k', linewidth=0.3)
        plt.title("UMAP: Latent Space Topology (Patch-based)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(OUTPUT_UMAP, dpi=200, bbox_inches="tight")
        print(f"Saved: {OUTPUT_UMAP}")
    except ImportError:
        print("UMAP skipped. Install: pip install umap-learn")


if __name__ == "__main__":
    main()