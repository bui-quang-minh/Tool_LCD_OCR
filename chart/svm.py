import os
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Config
TRAIN_IMAGES_DIR = Path(r"F:\clone\Tool_LCD_OCR\OCR_Model\train\images")
TRAIN_LABELS_DIR = Path(r"F:\clone\Tool_LCD_OCR\OCR_Model\train\labels")
PATCH_SIZE = 16
MAX_IMAGES = 700


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


def load_and_patch_images(image_dir, label_dir, patch_size=16):
    all_patches = []
    all_labels = []

    image_paths = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    print(f"Found {len(image_paths)} images. Extracting patches...")

    for i, img_path in enumerate(image_paths[:MAX_IMAGES]):
        gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        h, w = gray.shape[:2]
        label_path = label_dir / (img_path.stem + ".txt")
        polygons = parse_yolo_obb_label(label_path, w, h)

        h2, w2 = (h // patch_size) * patch_size, (w // patch_size) * patch_size

        for y in range(0, h2, patch_size):
            for x in range(0, w2, patch_size):
                patch = gray[y:y + patch_size, x:x + patch_size]
                vector = patch.ravel().astype(np.float32) / 255.0

                # Ground-truth label from OBB annotations
                overlap = patch_overlap_ratio(x, y, patch_size, polygons)
                if 0.10 <= overlap <= 0.70:
                    continue  # skip ambiguous patches
                label = 1 if overlap > 0.70 else 0

                all_patches.append(vector)
                all_labels.append(label)

        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{len(image_paths)} images...")

    return np.array(all_patches), np.array(all_labels)


def main():
    if not TRAIN_IMAGES_DIR.exists():
        print(f"Error: {TRAIN_IMAGES_DIR} not found")
        return

    X, y = load_and_patch_images(TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, PATCH_SIZE)
    print(f"Total patches: {X.shape[0]}")
    print(f"Background: {np.sum(y==0)}, Digit: {np.sum(y==1)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Linear SVM...")
    clf = LinearSVC(random_state=42, max_iter=10000, class_weight='balanced')
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print("-" * 40)
    print(f"Linear SVM ({len(X)} patches):")
    print(f"  Train Accuracy: {train_acc * 100:.2f}%")
    print(f"  Test Accuracy:  {test_acc * 100:.2f}%")
    print("-" * 40)
    print(classification_report(y_test, clf.predict(X_test),
                                target_names=["Background", "Digit"]))

    # --- Visualization ---
    print("Generating PCA visualization...")

    pca = PCA(n_components=2)
    X_test_2d = pca.fit_transform(X_test)
    var_expl = pca.explained_variance_ratio_.sum() * 100

    clf_2d = SVC(kernel='linear', C=1.0)
    clf_2d.fit(X_test_2d, y_test)

    plt.figure(figsize=(10, 8))

    m = 1.0
    xx, yy = np.meshgrid(
        np.linspace(X_test_2d[:,0].min()-m, X_test_2d[:,0].max()+m, 500),
        np.linspace(X_test_2d[:,1].min()-m, X_test_2d[:,1].max()+m, 500)
    )
    Z = clf_2d.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[-100, 0, 100], alpha=0.2, colors=['#D3D3D3', '#8B0000'])
    plt.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['-'], linewidths=2)

    plt.scatter(X_test_2d[y_test == 0, 0], X_test_2d[y_test == 0, 1],
                c='#D3D3D3', label='Background (0)', edgecolors='k', s=20, alpha=0.5)
    plt.scatter(X_test_2d[y_test == 1, 0], X_test_2d[y_test == 1, 1],
                c='#8B0000', label='Digit Segment (1)', edgecolors='k', s=20, alpha=0.7)

    plt.title(f"SVM Decision Boundary (Test Acc: {test_acc * 100:.2f}%)\n"
              f"PCA Variance Explained: {var_expl:.1f}%", fontsize=14)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)

    output_path = Path(__file__).parent / "svm_actual_result.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()