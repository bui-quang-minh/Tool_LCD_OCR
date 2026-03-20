"""
t-SNE and UMAP visualization for LCD digit vs background patches.
Uses YOLO OBB annotations as ground-truth labels.
"""
from pathlib import Path
import numpy as np
import cv2

IMAGES_DIR = Path(r"F:\clone\Tool_LCD_OCR\OCR_Model\train\images")
LABELS_DIR = Path(r"F:\clone\Tool_LCD_OCR\OCR_Model\train\labels")
CHART_DIR = Path(__file__).resolve().parent
PATCH_SIZE = 16
MAX_IMAGES = 700
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


def load_data(images_dir, labels_dir, patch_size):
    all_patches = []
    all_labels = []

    image_paths = list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} images.")

    for i, img_path in enumerate(image_paths[:MAX_IMAGES]):
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        h, w = gray.shape[:2]
        label_path = labels_dir / (img_path.stem + ".txt")
        polygons = parse_yolo_obb_label(label_path, w, h)

        h2 = (h // patch_size) * patch_size
        w2 = (w // patch_size) * patch_size

        for y in range(0, h2, patch_size):
            for x in range(0, w2, patch_size):
                overlap = patch_overlap_ratio(x, y, patch_size, polygons)
                if 0.10 <= overlap <= 0.70:
                    continue
                label = 1 if overlap > 0.70 else 0
                patch = gray[y:y+patch_size, x:x+patch_size]
                all_patches.append(patch.ravel().astype(np.float32) / 255.0)
                all_labels.append(label)

        if (i + 1) % 5 == 0:
            print(f"Processed {i+1}/{min(len(image_paths), MAX_IMAGES)}")

    return np.array(all_patches), np.array(all_labels)


def main():
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data(IMAGES_DIR, LABELS_DIR, PATCH_SIZE)
    print(f"Total patches: {X.shape[0]}")
    print(f"Background: {np.sum(y==0)}, Digit: {np.sum(y==1)}")

    if X.shape[0] < 2:
        print("Not enough data.")
        return

    colors = np.where(y == 1, '#8B0000', '#D3D3D3')

    # Linear SVM
    from sklearn.svm import LinearSVC
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LinearSVC(random_state=42, max_iter=10000, class_weight='balanced')
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"\nLinear SVM — Train: {train_acc*100:.2f}%  Test: {test_acc*100:.2f}%")
    print(classification_report(y_test, clf.predict(X_test),
                                target_names=["Background", "Digit"]))

    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    # t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0] - 1))
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 7))
    plt.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1],
                c='#D3D3D3', label='Background (0)', alpha=0.5, s=30, edgecolors='k', linewidth=0.3)
    plt.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1],
                c='#8B0000', label='Digit Segment (1)', alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
    plt.title(f"t-SNE: LCD Patch Separability (SVM Test Acc: {test_acc*100:.1f}%)")
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
        reducer = umap.UMAP(n_components=2, random_state=42,
                            n_neighbors=min(15, X.shape[0] - 1))
        X_umap = reducer.fit_transform(X)

        plt.figure(figsize=(10, 7))
        plt.scatter(X_umap[y==0, 0], X_umap[y==0, 1],
                    c='#D3D3D3', label='Background (0)', alpha=0.5, s=30, edgecolors='k', linewidth=0.3)
        plt.scatter(X_umap[y==1, 0], X_umap[y==1, 1],
                    c='#8B0000', label='Digit Segment (1)', alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
        plt.title(f"UMAP: LCD Patch Topology (SVM Test Acc: {test_acc*100:.1f}%)")
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