import cv2
import albumentations as A
import numpy as np
import os


def build_transform():
    # We handle bboxes manually via keypoints, so no bbox_params here.
    # Keypoint format lets us transform all 8 OBB corners correctly.
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
    Returns:
        class_labels: list of int
        polygons: list of shape (4, 2), normalized float coords
    """
    class_labels = []
    polygons = []

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            class_labels.append(int(float(parts[0])))
            coords = [float(x) for x in parts[1:9]]
            # Reshape into 4 (x, y) pairs
            poly = np.array(coords).reshape(4, 2)
            polygons.append(poly)

    return class_labels, polygons


def polygons_to_keypoints(polygons, img_w, img_h):
    """
    Flatten all polygon corners into a single keypoints list for Albumentations.
    Albumentations 'xy' keypoint format uses absolute pixel coordinates.
    We also need to track which keypoints belong to which polygon (4 points each).
    """
    keypoints = []
    for poly in polygons:
        for (nx, ny) in poly:
            # Convert normalized -> absolute pixel
            keypoints.append((nx * img_w, ny * img_h))
    return keypoints


def keypoints_to_polygons(keypoints, img_w, img_h, n_polygons):
    """
    Rebuild normalized polygons from the flat transformed keypoints list.
    Clips to [0, 1] to handle any floating point edge cases after transform.
    """
    polygons = []
    for i in range(n_polygons):
        pts = keypoints[i * 4: i * 4 + 4]
        poly = []
        for (px, py) in pts:
            nx = np.clip(px / img_w, 0.0, 1.0)
            ny = np.clip(py / img_h, 0.0, 1.0)
            poly.append((nx, ny))
        polygons.append(poly)
    return polygons


def is_valid_polygon(poly):
    """
    Reject polygons that have collapsed to a point or line after transform,
    which happens when the object was mostly outside the image boundary.
    A minimum area threshold avoids saving degenerate annotations.
    """
    pts = np.array(poly, dtype=np.float32)
    # Shoelace formula for polygon area
    x = pts[:, 0]
    y = pts[:, 1]
    area = 0.5 * abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )
    return area > 1e-4  # normalized area threshold


def save_obb_label(label_path, class_labels, polygons):
    with open(label_path, 'w') as f:
        for cls, poly in zip(class_labels, polygons):
            coords = ' '.join(f'{v:.6f}' for pt in poly for v in pt)
            f.write(f"{cls} {coords}\n")


def augment_training_data():
    transform = build_transform()

    input_dir = "trainobb(base)"
    output_dir = "trainobb"

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    image_files = [
        f for f in os.listdir(f"{input_dir}/images")
        if f.endswith(('.jpg', '.png'))
    ]

    print(f"Starting augmentation on {len(image_files)} images...")

    for img_file in image_files:
        img_path = os.path.join(input_dir, "images", img_file)
        label_file = img_file.rsplit('.', 1)[0] + ".txt"
        label_path = os.path.join(input_dir, "labels", label_file)

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_h, img_w = img.shape[:2]

        class_labels, polygons = parse_obb_label(label_path)

        if not polygons:
            continue

        # Save the original unchanged
        cv2.imwrite(
            f"{output_dir}/images/{img_file}",
            cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        )
        save_obb_label(f"{output_dir}/labels/{label_file}", class_labels, polygons)

        # Generate augmented versions
        for i in range(3):
            try:
                keypoints = polygons_to_keypoints(polygons, img_w, img_h)
                result = transform(image=img, keypoints=keypoints)

                aug_img = result['image']
                aug_h, aug_w = aug_img.shape[:2]

                aug_polygons = keypoints_to_polygons(
                    result['keypoints'], aug_w, aug_h, len(polygons)
                )

                # Filter out any polygons that collapsed after transform
                valid = [
                    (cls, poly)
                    for cls, poly in zip(class_labels, aug_polygons)
                    if is_valid_polygon(poly)
                ]

                if not valid:
                    continue

                aug_labels, aug_polys = zip(*valid)

                base_name = img_file.rsplit('.', 1)[0]
                aug_name = f"{base_name}_aug_{i}"

                cv2.imwrite(
                    f"{output_dir}/images/{aug_name}.jpg",
                    cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                )
                save_obb_label(
                    f"{output_dir}/labels/{aug_name}.txt",
                    list(aug_labels),
                    list(aug_polys)
                )

            except Exception as e:
                print(f"Skipping aug {i} for {img_file}: {e}")

    print(f"\nDone! Check your {output_dir} folder.")


if __name__ == '__main__':
    augment_training_data()