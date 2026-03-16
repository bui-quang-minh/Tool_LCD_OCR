"""
IPV - Tool & LCD detection app.
Option 1: Run YOLO on validation set.
Option 2: Load image (file dialog or drag-and-drop).
Shows bounding boxes, LCD crop, and OCR via model in OCR_Model (with bbox overlay).
"""
import os
import re
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import numpy as np

# Project paths (prefer OBB model from runs folder)
PROJECT_ROOT = Path(__file__).resolve().parent
_OBB_PATHS = [
    PROJECT_ROOT / "runs" / "runs" / "obb" / "train" / "weights" / "best.pt",
    PROJECT_ROOT / "runs" / "obb" / "train" / "weights" / "best.pt",
]
MODEL_PATH = next((p for p in _OBB_PATHS if p.exists()), PROJECT_ROOT / r"runs\detect\train10\weights\best.pt")
VAL_DIR = PROJECT_ROOT / "val" / "images"
if not Path(VAL_DIR).exists():
    VAL_DIR = PROJECT_ROOT / "testfile"
    #VAL_DIR = PROJECT_ROOT / "trainobb(base)" / "images"
RAW_IMAGE_DIR = PROJECT_ROOT / "Raw Image"
OCR_MODEL_DIR = PROJECT_ROOT / "OCR_Model"
# New run (100 epochs, imgsz=320): OCR_Model/runs/obb/train/weights/best.pt
OCR_MODEL_WEIGHTS = OCR_MODEL_DIR / "runs" / "obb" / "train" / "weights" / "best.pt"
# OCR_MODEL_WEIGHTS = OCR_MODEL_DIR / "runs" / "obb" / "train3" / "weights" / "best.pt"  # old
# OCR model class names (digits + symbols)
OCR_CLASS_NAMES = [".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "nm", "mp"]
# Black border padding around crop for OCR so edge letters are not cut off
OCR_PADDING = 24
# Default rotation when OBB angle not available (e.g. 48 for your calibration view)
DEFAULT_ROTATION_DEG = 48.0

# Class names (from dataset.yaml)
CLASS_NAMES = ["lcd_display", "torque_wrench"]
LCD_CLASS_ID = 0


def get_val_images():
    """Return list of image paths in val folder (match labels by base name)."""
    val_path = Path(VAL_DIR)
    if not val_path.exists():
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = []
    for ext in exts:
        images.extend(val_path.glob(f"*{ext}"))
    labels_dir = val_path / "labels"
    if labels_dir.exists():
        # Also allow images in val root with same base name as label files
        for lb in labels_dir.glob("*.txt"):
            base = lb.stem
            for ext in exts:
                p = val_path / f"{base}{ext}"
                if p.exists() and p not in images:
                    images.append(p)
    return sorted(set(images))


def get_raw_images():
    """Return list of image paths in Raw Image folder."""
    raw_path = Path(RAW_IMAGE_DIR)
    if not raw_path.exists():
        return []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    images = []
    for ext in exts:
        images.extend(raw_path.glob(f"*{ext}"))
    return sorted(set(images))


def load_yolo():
    """Load YOLO model."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        return model
    except Exception as e:
        messagebox.showerror("Model load error", str(e))
        return None


def load_ocr_model():
    """Load OCR YOLO model from OCR_Model folder. Returns None if not found or error."""
    if not OCR_MODEL_WEIGHTS.exists():
        return None
    try:
        from ultralytics import YOLO
        return YOLO(str(OCR_MODEL_WEIGHTS))
    except Exception:
        return None


# Expected OCR format: xx.xnm (2 digits, decimal, 1 digit, "nm" or "mp")
OCR_FORMAT_RE = re.compile(r"^\d{2}\.\d{1}(?:nm|mp)$")


def _ocr_validate_format(text):
    """Return True if text matches xx.xnm or xx.xmp."""
    if not text:
        return False
    return bool(OCR_FORMAT_RE.match(text.strip().lower()))


def _ocr_fix_format(raw):
    """Try to fix raw OCR string to xx.xnm. Returns (display_string, was_fixed) or (None, False) if not fixable."""
    if not raw:
        return None, False
    digits_only = "".join(c for c in raw if c.isdigit())
    s = "".join(c for c in raw if c.isdigit() or c == ".")
    # Already xx.xnm or xx.xmp
    if _ocr_validate_format(raw.strip()):
        out = raw.strip().lower()
        return (out.replace("mp", "nm"), False)
    # Already xx.x (2 digits, dot, 1 digit) -> add nm
    if len(s) == 4 and s[2] == "." and s[:2].isdigit() and s[3].isdigit():
        return s + "nm", True
    # Exactly 3 digits -> xx.xnm (e.g. 511 -> 51.1nm)
    if len(digits_only) == 3:
        return f"{digits_only[0]}{digits_only[1]}.{digits_only[2]}nm", True
    # 4+ digits: use first two + last (decimal digit). Avoids 6667 -> 66.6; use 66.7 when order/dup gives extra digit
    if len(digits_only) >= 3:
        return f"{digits_only[0]}{digits_only[1]}.{digits_only[-1]}nm", True
    # One dot: try xx.x (e.g. "51.1" -> 51.1nm)
    if "." in s:
        parts = s.split(".")
        if len(parts) == 2 and len(parts[0]) >= 2 and parts[1]:
            a, b = parts[0][-2:], parts[1][0] if parts[1] else "0"
            return f"{a}.{b}nm", True
    return None, False


# Filter names for status (same order as _ocr_apply_try_filter)
OCR_FILTER_NAMES = [
    "Original", "Grayscale", "Binary (adaptive)", "Grayscale + Resize 2x",
    "Border (morph gradient)", "Invert", "Contrast",
]


def _ocr_clean_display(display_text):
    """Strip debug suffixes for OCR result box: ' (fixed)', ' (N tried)'."""
    if not display_text or not isinstance(display_text, str):
        return display_text
    return display_text.split(" (")[0].strip()


def _ocr_format_with_retries(ocr_model, image_bgr, max_tries=7):
    """Run OCR up to max_tries times; validate/fix to xx.xnm. Returns (display_string, vis, n_tries, filter_index, raw_string)."""
    best_fixed = None
    best_was_fixed = False
    best_vis = None
    best_attempt = 0
    best_raw = ""
    last_vis = None
    last_raw = ""
    for attempt in range(max_tries):
        filtered = _ocr_apply_try_filter(image_bgr, attempt)
        vis, raw = run_ocr_model_annotated(ocr_model, filtered)
        if vis is not None:
            last_vis = vis
        last_raw = raw or ""
        fixed, was_fixed = _ocr_fix_format(last_raw)
        if fixed and _ocr_validate_format(fixed):
            display = f"{fixed} (fixed)" if was_fixed else fixed
            return display, last_vis, attempt + 1, attempt, last_raw
        if fixed:
            best_fixed = fixed
            best_was_fixed = was_fixed
            best_vis = last_vis
            best_attempt = attempt
            best_raw = last_raw
    if best_fixed is not None:
        return f"{best_fixed} (fixed) ({max_tries} tried)", (best_vis if best_vis is not None else last_vis), max_tries, best_attempt, best_raw
    if not last_raw or not any(c.isdigit() for c in last_raw):
        return "(can not read image)", last_vis, max_tries, -1, last_raw
    return f"{last_raw.strip()} ({max_tries} tried)", last_vis, max_tries, -1, last_raw


def parse_nm_from_ocr_text(display_text):
    """Extract numeric value in Nm from OCR result string. Returns float or None if not parseable."""
    if not display_text or not isinstance(display_text, str):
        return None
    m = re.search(r"(\d+\.?\d*)", display_text.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def is_nm_in_range(value_nm, low=60.0, high=80.0):
    """Return True if value is in [low, high] Nm (inclusive)."""
    if value_nm is None:
        return False
    return low <= value_nm <= high


def add_ocr_padding(image_bgr_or_gray, padding=None):
    """Add a black border around the image so edge characters are not cut off. Returns BGR image or None."""
    if image_bgr_or_gray is None or getattr(image_bgr_or_gray, "size", 0) == 0:
        return None
    if padding is None:
        padding = OCR_PADDING
    img = image_bgr_or_gray
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    new_h, new_w = h + 2 * padding, w + 2 * padding
    out = np.zeros((new_h, new_w, 3), dtype=img.dtype)
    out[:] = (0, 0, 0)
    out[padding : padding + h, padding : padding + w] = img
    return out


def _ocr_padded_or_original(image_bgr_or_gray):
    """Return padded image for OCR, or original if padding fails. Never use 'or' with numpy arrays."""
    padded = add_ocr_padding(image_bgr_or_gray)
    if padded is None:
        return image_bgr_or_gray
    return padded


def _ocr_apply_try_filter(image_bgr, try_index):
    """Apply a different filter for each OCR retry. Kept: 0,1,3,5,8,9 + border (kernel). Returns BGR image for OCR."""
    if image_bgr is None or getattr(image_bgr, "size", 0) == 0:
        return image_bgr
    filters = [
        lambda im: im,  # 0: original
        lambda im: cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),  # 1: grayscale
        lambda im: _ocr_bgr_to_binary_adaptive(im),  # 3: binary adaptive
        lambda im: cv2.cvtColor(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR),  # 5: grayscale + resize 2x
        lambda im: _ocr_bgr_to_border(im),  # border: morphological gradient (kernel)
        lambda im: _ocr_invert(im),  # 8: invert
        lambda im: _ocr_contrast(im),  # 9: contrast
    ]
    idx = try_index % len(filters)
    out = filters[idx](image_bgr)
    return out if out is not None and getattr(out, "size", 0) > 0 else image_bgr


def _ocr_bgr_to_binary_otsu(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _ocr_bgr_to_binary_adaptive(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(g, (3, 3), 0)
    out = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _ocr_morph_open(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    out = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _ocr_invert(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = cv2.bitwise_not(g)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _ocr_contrast(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _ocr_bgr_to_border(bgr):
    """Kernel-based filter: morphological gradient to draw detected borders (dilate - erode)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    out = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, kernel)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _ocr_dedupe_items(items, img_width, min_gap_frac=0.03):
    """Merge only detections that are very close in x (same character detected twice); keep higher confidence. Too high and the '.' is merged with digits."""
    min_gap = max(4, img_width * min_gap_frac)
    dedup = []
    for t in items:
        x = t[0]
        conf = t[2] if len(t) >= 3 else 1.0
        if dedup and abs(x - dedup[-1][0]) < min_gap:
            if len(dedup[-1]) >= 3 and conf > dedup[-1][2]:
                dedup[-1] = t
        else:
            dedup.append(t)
    return dedup


def run_ocr_model(ocr_model, image_bgr_or_gray):
    """Run OCR YOLO (OBB) on warped crop; return decoded string (left-to-right)."""
    if ocr_model is None or image_bgr_or_gray is None or getattr(image_bgr_or_gray, "size", 0) == 0:
        return ""
    img = image_bgr_or_gray
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    try:
        results = ocr_model(img, verbose=False)
        r = results[0] if results else None
        if r is None or not hasattr(r, "obb") or r.obb is None or len(r.obb) == 0:
            return ""
        w = img.shape[1]
        items = []
        for i in range(len(r.obb)):
            cls_id = int(r.obb.cls[i].item())
            conf = float(r.obb.conf[i].item())
            pts = r.obb.xyxyxyxy[i].cpu().numpy() if hasattr(r.obb.xyxyxyxy[i], "cpu") else r.obb.xyxyxyxy[i]
            corners = np.array(pts, dtype=np.float64).reshape(4, 2)
            center_x = float(corners[:, 0].mean())
            items.append((center_x, cls_id, conf))
        items.sort(key=lambda x: x[0])
        items = _ocr_dedupe_items(items, w)
        names = OCR_CLASS_NAMES
        return "".join(names[cid] if 0 <= cid < len(names) else "?" for _, cid, *_ in items)
    except Exception as e:
        return f"OCR error: {e}"


def run_ocr_model_annotated(ocr_model, image_bgr_or_gray):
    """Run OCR YOLO (OBB) on image; return (annotated BGR image with bboxes, decoded string)."""
    if ocr_model is None or image_bgr_or_gray is None or getattr(image_bgr_or_gray, "size", 0) == 0:
        return None, ""
    img = image_bgr_or_gray
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    vis = img.copy()
    try:
        results = ocr_model(img, verbose=False)
        r = results[0] if results else None
        if r is None or not hasattr(r, "obb") or r.obb is None or len(r.obb) == 0:
            return vis, ""
        names = OCR_CLASS_NAMES
        items = []
        h, w = img.shape[:2]
        for i in range(len(r.obb)):
            cls_id = int(r.obb.cls[i].item())
            conf = float(r.obb.conf[i].item())
            pts = r.obb.xyxyxyxy[i].cpu().numpy() if hasattr(r.obb.xyxyxyxy[i], "cpu") else r.obb.xyxyxyxy[i]
            corners = np.array(pts, dtype=np.int32).reshape(4, 2)
            center_x = float(corners[:, 0].mean())
            items.append((center_x, cls_id, conf, corners))
        items.sort(key=lambda x: x[0])
        items = _ocr_dedupe_items(items, w)
        text = "".join(names[cid] if 0 <= cid < len(names) else "?" for _, cid, _, _ in items)
        for _, cls_id, _, corners in items:
            label = names[cls_id] if 0 <= cls_id < len(names) else "?"
            cv2.polylines(vis, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            # Draw label under the box so it won't overlap
            bottom_y = int(corners[:, 1].max())
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cx = int(corners[:, 0].mean())
            tx = cx - tw // 2
            ty = bottom_y + th + 4
            cv2.putText(vis, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return vis, text
    except Exception as e:
        return vis, f"OCR error: {e}"


def run_inference(model, image_bgr):
    """Run YOLO on BGR image; return (annotated BGR image, list of detections).
    Detections: OBB mode: (class_id, conf, corners_4, angle_deg); else (class_id, x1, y1, x2, y2, conf).
    """
    results = model(image_bgr, verbose=False)
    detections = []
    out = image_bgr.copy()
    r = results[0] if results else None
    # OBB model returns result.obb with xyxyxyxy (4 corners) and angle in xywhr
    if r is not None and hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
        for i in range(len(r.obb)):
            cls_id = int(r.obb.cls[i].item())
            conf = float(r.obb.conf[i].item())
            # xyxyxyxy: 4 corners (x1,y1, x2,y2, x3,y3, x4,y4)
            pts = r.obb.xyxyxyxy[i].cpu().numpy() if hasattr(r.obb.xyxyxyxy[i], "cpu") else r.obb.xyxyxyxy[i]
            corners = np.array(pts, dtype=np.int32).reshape(4, 2)
            # Angle from xywhr (x, y, w, h, r); ultralytics may use radians or degrees
            xywhr = r.obb.xywhr[i].cpu().numpy() if hasattr(r.obb.xywhr[i], "cpu") else r.obb.xywhr[i]
            r_val = float(xywhr[4])
            angle_deg = np.degrees(r_val) if abs(r_val) <= np.pi + 0.1 else r_val
            detections.append((cls_id, conf, corners, angle_deg))
            color = (0, 255, 0) if cls_id == LCD_CLASS_ID else (255, 165, 0)
            cv2.polylines(out, [corners], isClosed=True, color=color, thickness=2)
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            cv2.putText(out, label, (int(corners[0][0]), int(corners[0][1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((cls_id, x1, y1, x2, y2, conf))
                color = (0, 255, 0) if cls_id == LCD_CLASS_ID else (255, 165, 0)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                cv2.putText(out, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out, detections


def _is_obb_detection(det):
    """True if det is (class_id, conf, corners_4, angle_deg)."""
    if len(det) != 4:
        return False
    c = np.asarray(det[2])
    return c.shape == (4, 2)


def _default_rotation_for_crop(crop):
    """Default rotation: 90° if shorter side is upright (portrait), else 180°."""
    if crop is None or crop.size == 0:
        return 180.0
    h, w = crop.shape[:2]
    # Portrait = height > width (shorter side horizontal, longer side upright)
    return 90.0 if h > w else 180.0


def _order_quad_tl_tr_br_bl(corners):
    """Order 4 corners as [top-left, top-right, bottom-right, bottom-left] so warp is never mirrored.
    OBB can return corners in varying order; use angle-from-centroid + tl = min(x+y) and enforce
    tl->tr->br->bl (tr has larger x than tl) to avoid random mirroring.
    """
    pts = np.asarray(corners, dtype=np.float32)
    if pts.shape != (4, 2):
        return pts
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    # Sort by angle around centroid for consistent cyclic order (counter-clockwise in image y-down)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    ordered = pts[order]
    # tl = vertex with smallest x+y (top-left in image coords)
    sums = ordered[:, 0] + ordered[:, 1]
    tl_pos = int(np.argmin(sums))
    # Rotate so tl is first: [tl, next, next, next]
    rotated = np.roll(ordered, -tl_pos, axis=0)
    # Ensure winding tl -> tr -> br -> bl: after tl, tr must have larger x than tl (else we have tl->bl and are mirrored)
    if rotated[1][0] <= rotated[0][0]:
        rotated[1:4] = rotated[3:0:-1]
    return np.array(rotated, dtype=np.float32)


def crop_lcd_regions(image_bgr, detections, class_id=LCD_CLASS_ID):
    """Return (list of BGR crops, list of angles_deg for straightening).
    OBB: crop by warping quad to rect, angle stored for init rotation. Axis-aligned: angle 0.
    """
    h, w = image_bgr.shape[:2]
    crops = []
    angles = []
    for det in detections:
        if _is_obb_detection(det):
            cid, conf, corners, angle_deg = det
            if cid != class_id:
                continue
            # Order corners tl, tr, br, bl so the crop is never mirrored
            src = _order_quad_tl_tr_br_bl(corners)
            w1 = np.linalg.norm(src[1] - src[0])
            w2 = np.linalg.norm(src[2] - src[3])
            h1 = np.linalg.norm(src[3] - src[0])
            h2 = np.linalg.norm(src[2] - src[1])
            dw = int(max(10, (w1 + w2) / 2))
            dh = int(max(10, (h1 + h2) / 2))
            dst = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            crop = cv2.warpPerspective(image_bgr, M, (dw, dh), flags=cv2.INTER_LINEAR)
            crops.append(crop)
            angles.append(angle_deg)
        else:
            cid, x1, y1, x2, y2, _ = det
            if cid != class_id:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crops.append(image_bgr[y1:y2, x1:x2].copy())
                angles.append(0.0)
    return crops, angles


def apply_rotation(crop_bgr, angle_deg):
    """Rotate crop by angle_deg (degrees). Expands canvas so nothing is cut off."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if abs(angle_deg) < 0.01:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nw = int((h * sin + w * cos))
    nh = int((h * cos + w * sin))
    M[0, 2] += (nw / 2.0) - cx
    M[1, 2] += (nh / 2.0) - cy
    return cv2.warpAffine(crop_bgr, M, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def apply_perspective(crop_bgr, src_quad):
    """Warp crop from quadrilateral src_quad to a rectangle. src_quad: 4 (x,y) [tl, tr, br, bl]."""
    if crop_bgr is None or crop_bgr.size == 0 or src_quad is None or len(src_quad) != 4:
        return crop_bgr
    h, w = crop_bgr.shape[:2]
    src = np.array(src_quad, dtype=np.float32)
    # Output size from quad: use mean of opposite sides
    w1 = np.linalg.norm(src[1] - src[0])
    w2 = np.linalg.norm(src[2] - src[3])
    h1 = np.linalg.norm(src[3] - src[0])
    h2 = np.linalg.norm(src[2] - src[1])
    dw = int(max(10, (w1 + w2) / 2))
    dh = int(max(10, (h1 + h2) / 2))
    dst = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(crop_bgr, M, (dw, dh), flags=cv2.INTER_LINEAR)


def process_crop_for_export(crop_bgr, angle_deg, transform_name):
    """Process a single LCD crop for export: full rect -> rotate -> transform. Returns BGR or gray array."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    quad = [(0, 0), (w, 0), (w, h), (0, h)]
    out = apply_perspective(crop_bgr, quad)
    if out is None:
        out = crop_bgr
    out = apply_rotation(out, angle_deg)
    if out is None:
        out = crop_bgr
    out = apply_transform(out, transform_name)
    return out


def apply_transform(crop_bgr, transform_name):
    """Apply transform for OCR: grayscale, threshold, resize, etc."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    if transform_name == "None":
        return crop_bgr
    if transform_name == "Grayscale":
        return cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if transform_name == "Binary (Otsu)":
        g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        _, out = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return out
    if transform_name == "Binary (adaptive)":
        g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    if transform_name == "Resize 2x":
        return cv2.resize(crop_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    if transform_name == "Grayscale + Resize 2x":
        g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return crop_bgr


def cv2_to_photoimage(bgr_or_gray):
    """Convert OpenCV image to ImageTk.PhotoImage for tkinter."""
    if bgr_or_gray is None or bgr_or_gray.size == 0:
        return None
    if len(bgr_or_gray.shape) == 2:
        pil = Image.fromarray(bgr_or_gray)
    else:
        pil = Image.fromarray(cv2.cvtColor(bgr_or_gray, cv2.COLOR_BGR2RGB))
    return ImageTk.PhotoImage(pil)


def fit_image_to_label(cv_img, max_w, max_h):
    """Scale image to fit within max_w x max_h keeping aspect ratio."""
    if cv_img is None or cv_img.size == 0:
        return None
    h, w = cv_img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        cv_img = cv2.resize(cv_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return cv_img


class IPVApp:
    def __init__(self, root=None):
        self.root = root if root is not None else tk.Tk()
        self.root.title("IPV - Tool & LCD Detection")
        self.root.geometry("1200x800")
        self.root.minsize(900, 600)

        self.model = None
        self.ocr_model = None
        self.current_image_bgr = None
        self.current_annotated = None
        self.current_detections = []
        self.lcd_crops = []
        self.lcd_crop_angles = []  # per-crop angle (deg) from OBB for init straightening
        self.current_crop_index = 0
        self.val_image_paths = []
        self.val_index = 0
        self.photo_main = None
        self.photo_crop = None
        self.rotate_angle = 0.0
        self.flip_h = False
        self.flip_v = False
        self._ocr_vis_photo = None  # photo for OCR result canvas
        self._ocr_after_id = None   # debounce: run OCR auto after crop display updates

        self._build_ui()
        self._load_model()

    def _load_model(self):
        self.model = load_yolo()
        self.ocr_model = load_ocr_model()
        if self.model is not None:
            self.status_var.set("Model loaded." + (" OCR model loaded." if self.ocr_model else " (OCR model not found.)"))
        else:
            self.status_var.set("Failed to load model.")

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Ready.")

        # --- Top: Mode and actions ---
        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(top, text="Mode:", font=("Segoe UI", 10)).pack(side=tk.LEFT, padx=(0, 8))
        self.mode_var = tk.StringVar(value="val")
        ttk.Radiobutton(top, text="1. Run test on val set", variable=self.mode_var, value="val", command=self._on_mode_change).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(top, text="2. Load image (file / drag-drop)", variable=self.mode_var, value="file", command=self._on_mode_change).pack(side=tk.LEFT, padx=4)
        ttk.Separator(top, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=12)
        self.btn_run_val = ttk.Button(top, text="Load next val image", command=self._run_val)
        self.btn_run_val.pack(side=tk.LEFT, padx=4)
        self.btn_open = ttk.Button(top, text="Open image...", command=self._open_image)
        self.btn_open.pack(side=tk.LEFT, padx=4)
        self.btn_drop_zone = ttk.Label(top, text="  Drop image here  ", relief=tk.RIDGE, padding=6, cursor="hand2")
        self.btn_drop_zone.pack(side=tk.LEFT, padx=4)
        self.btn_drop_zone.bind("<Button-1>", lambda e: self._open_image())
        self._on_mode_change()

        # --- Content: image + crop + OCR ---
        content = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        content.pack(fill=tk.BOTH, expand=True, pady=4)

        # Left: main image with bboxes
        left = ttk.Frame(content)
        content.add(left, weight=2)
        ttk.Label(left, text="Image (with bounding boxes)", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.canvas_main = tk.Canvas(left, bg="#2b2b2b", highlightthickness=0)
        self.canvas_main.pack(fill=tk.BOTH, expand=True)
        self.canvas_main.bind("<Configure>", self._on_canvas_resize)

        # Right: LCD crop + transform + OCR
        right = ttk.Frame(content)
        content.add(right, weight=1)
        ttk.Label(right, text="LCD crop (for OCR)", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        self.frame_crop = ttk.Frame(right)
        self.frame_crop.pack(fill=tk.BOTH, expand=True)
        self.canvas_crop = tk.Canvas(self.frame_crop, bg="#1e1e1e", highlightthickness=0, width=320, height=240)
        self.canvas_crop.pack(fill=tk.BOTH, expand=True)

        # Rotate
        rot_frame = ttk.Frame(right)
        rot_frame.pack(fill=tk.X, pady=(8, 0))
        ttk.Label(rot_frame, text="Rotate (°):", font=("Segoe UI", 9)).pack(side=tk.LEFT, padx=(0, 4))
        self.rotate_var = tk.DoubleVar(value=0.0)
        self.rotate_scale = ttk.Scale(rot_frame, from_=-180, to=180, variable=self.rotate_var, orient=tk.HORIZONTAL, length=120, command=lambda v: self._on_rotate_change())
        self.rotate_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self.rotate_label = ttk.Label(rot_frame, text="0", font=("Segoe UI", 9), width=4)
        self.rotate_label.pack(side=tk.LEFT)
        ttk.Button(rot_frame, text="0°", width=3, command=lambda: self._set_rotate(0)).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_frame, text="+90°", width=4, command=self._rotate_90_cw).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_frame, text="-90°", width=4, command=self._rotate_90_ccw).pack(side=tk.LEFT, padx=2)
        ttk.Button(rot_frame, text="Default", width=5, command=self._set_rotate_default).pack(side=tk.LEFT, padx=2)

        # OCR_Model result: LCD crop (same angle) with OCR bboxes drawn
        ttk.Label(right, text="OCR_Model (bounding boxes):", font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(8, 0))
        self.ocr_result_canvas = tk.Canvas(right, bg="#1a1a1a", highlightthickness=1, highlightbackground="#444", width=300, height=160)
        self.ocr_result_canvas.pack(fill=tk.X, pady=2)

        ttk.Label(right, text="Transform:", font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(8, 0))
        self.transform_var = tk.StringVar(value="None")
        transforms = ["None", "Grayscale", "Binary (Otsu)", "Binary (adaptive)", "Resize 2x", "Grayscale + Resize 2x"]
        cb = ttk.Combobox(right, textvariable=self.transform_var, values=transforms, state="readonly", width=22)
        cb.pack(fill=tk.X, pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_crop_display())
        ttk.Button(right, text="Run OCR (OCR_Model)", command=self._run_ocr_model).pack(pady=8)
        # ttk.Button(right, text="Export all detected to OCR_Model", command=self._export_all_to_ocr_model).pack(pady=4)
        ttk.Label(right, text="OCR result:", font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(4, 0))
        ocr_result_row = ttk.Frame(right)
        ocr_result_row.pack(fill=tk.X, pady=4)
        # Left: text result
        ocr_left = ttk.Frame(ocr_result_row)
        ocr_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.ocr_text = tk.Text(ocr_left, height=6, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.ocr_text.pack(fill=tk.BOTH, expand=True)
        # Right: OK (green) / NG (red) for 60–80 Nm
        ocr_right = ttk.Frame(ocr_result_row)
        ocr_right.pack(side=tk.RIGHT, padx=(8, 0), fill=tk.Y)
        self.ocr_verdict_label = tk.Label(
            ocr_right, text="—", font=("Segoe UI", 14, "bold"), width=4,
            relief=tk.RIDGE, padx=8, pady=8, bg="#333", fg="#aaa",
        )
        self.ocr_verdict_label.pack(expand=True)

        # Status
        ttk.Label(main, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, pady=(4, 0))

        # Drag and drop is enabled in main() when tkinterdnd2 is available

    def _get_processed_crop(self, crop):
        """Apply rotate -> flip -> transform to get final crop for display/OCR (LCD crop with same angle)."""
        if crop is None or crop.size == 0:
            return None
        out = apply_rotation(crop, self.rotate_var.get())
        if out is None:
            out = crop
        if self.flip_h:
            out = cv2.flip(out, 1)
        if self.flip_v:
            out = cv2.flip(out, 0)
        out = apply_transform(out, self.transform_var.get())
        return out if (out is not None and getattr(out, "size", 0) > 0) else crop

    def _on_rotate_change(self, *_):
        self.rotate_label.config(text=f"{int(round(self.rotate_var.get()))}")
        self._refresh_crop_display()

    def _set_rotate(self, angle):
        self.rotate_var.set(float(angle))
        self.rotate_label.config(text=f"{int(angle)}")
        self._refresh_crop_display()

    def _set_rotate_default(self):
        """Set rotation to the saved default (e.g. 48°)."""
        global DEFAULT_ROTATION_DEG
        self.rotate_var.set(float(DEFAULT_ROTATION_DEG))
        self.rotate_label.config(text=f"{int(round(DEFAULT_ROTATION_DEG))}")
        self._refresh_crop_display()

    def _set_current_as_default(self):
        """Save current rotation as the default for next load."""
        global DEFAULT_ROTATION_DEG
        DEFAULT_ROTATION_DEG = float(self.rotate_var.get())
        self.status_var.set(f"Default rotation set to {int(round(DEFAULT_ROTATION_DEG))}°.")

    def _rotate_90_cw(self):
        a = (self.rotate_var.get() + 90) % 360
        if a > 180:
            a -= 360
        self.rotate_var.set(float(a))
        self.rotate_label.config(text=f"{int(round(a))}")
        self._refresh_crop_display()

    def _rotate_90_ccw(self):
        a = (self.rotate_var.get() - 90) % 360
        if a > 180:
            a -= 360
        self.rotate_var.set(float(a))
        self.rotate_label.config(text=f"{int(round(a))}")
        self._refresh_crop_display()

    def _flip_h(self):
        self.flip_h = not self.flip_h
        self._refresh_crop_display()

    def _flip_v(self):
        self.flip_v = not self.flip_v
        self._refresh_crop_display()

    def _auto_fix_orientation(self):
        """Try rotations 0, 90, 180, 270 and flip H; pick orientation that gives best OCR (OCR_Model) result."""
        if not self.lcd_crops:
            return
        if self.ocr_model is None:
            self.ocr_model = load_ocr_model()
        if self.ocr_model is None:
            self.status_var.set("OCR model not loaded; cannot auto-fix.")
            return
        crop = self.lcd_crops[self.current_crop_index]
        if crop is None or crop.size == 0:
            return
        base = crop
        best_text = ""
        best_score = -1
        best_rot = 0.0
        best_fh = False
        for rot in (0, 90, 180, 270):
            for flip_h in (False, True):
                out = apply_rotation(base, float(rot))
                if out is None:
                    out = base
                if flip_h:
                    out = cv2.flip(out, 1)
                out = apply_transform(out, self.transform_var.get())
                if out is None or out.size == 0:
                    continue
                text = run_ocr_model(self.ocr_model, out)
                digits = "".join(c for c in text if c.isdigit() or c == ".")
                score = len(digits) + (10 if "." in digits else 0) + (5 if re.match(r"^\d+\.?\d*$", digits) else 0)
                if score > best_score:
                    best_score = score
                    best_text = text
                    best_rot = rot
                    best_fh = flip_h
        if best_score < 0:
            return
        self.rotate_var.set(float(best_rot))
        self.rotate_label.config(text=f"{int(best_rot)}")
        self.flip_h = best_fh
        self._refresh_crop_display()
        self.status_var.set(f"Auto-fix: rotation {int(best_rot)}°, flip_h={best_fh}. OCR: {best_text}")

    def _refresh_ocr_result_canvas(self, annotated_bgr):
        """Display OCR result image (crop with bboxes) in the OCR result canvas."""
        self.ocr_result_canvas.delete("all")
        if annotated_bgr is None or getattr(annotated_bgr, "size", 0) == 0:
            return
        cw = self.ocr_result_canvas.winfo_width() or 300
        ch = self.ocr_result_canvas.winfo_height() or 160
        fitted = fit_image_to_label(annotated_bgr, cw, ch)
        if fitted is not None:
            self._ocr_vis_photo = cv2_to_photoimage(fitted)
            self.ocr_result_canvas.create_image(cw // 2, ch // 2, image=self._ocr_vis_photo, tags="ocr_vis")

    def _on_mode_change(self):
        if self.mode_var.get() == "val":
            self.btn_run_val.pack(side=tk.LEFT, padx=4)
            self.btn_open.pack(side=tk.LEFT, padx=4)
            self.val_image_paths = get_val_images()
            self.val_index = 0
            self.status_var.set(f"Val mode. Found {len(self.val_image_paths)} images in val folder.")
        else:
            self.btn_run_val.pack_forget()
            self.status_var.set("Load an image via 'Open image...' or drag and drop.")

    def _on_canvas_resize(self, event):
        self._refresh_main_display()

    def _run_val(self):
        if not self.val_image_paths:
            self.val_image_paths = get_val_images()
            self.val_index = 0
        if not self.val_image_paths:
            messagebox.showinfo("Val set", "No images found in val folder. Add images (e.g. .jpg, .png) to:\n" + str(VAL_DIR))
            return
        path = self.val_image_paths[self.val_index % len(self.val_image_paths)]
        self.val_index += 1
        self._load_and_run(path)
        self.status_var.set(f"Val image {self.val_index}/{len(self.val_image_paths)}: {path.name}")

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp"), ("All", "*.*")]
        )
        if path:
            self._load_and_run(path)

    def _load_and_run(self, path):
        path = Path(path)
        if not path.exists():
            messagebox.showerror("Error", "File not found.")
            return
        img = cv2.imread(str(path))
        if img is None:
            messagebox.showerror("Error", "Could not load image.")
            return
        self.current_image_bgr = img
        self._run_detection()

    def _run_detection(self):
        if self.model is None or self.current_image_bgr is None:
            return
        self.current_annotated, self.current_detections = run_inference(self.model, self.current_image_bgr)
        self.lcd_crops, self.lcd_crop_angles = crop_lcd_regions(self.current_image_bgr, self.current_detections, LCD_CLASS_ID)
        self.current_crop_index = 0
        self.flip_h = False
        self.flip_v = False
        # Default rotation: 180° normally; 90° if shorter side is upright (portrait crop)
        init_angle = _default_rotation_for_crop(self.lcd_crops[0]) if self.lcd_crops else 180.0
        self.rotate_var.set(float(init_angle))
        self.rotate_label.config(text=f"{int(round(init_angle))}")
        self._refresh_main_display()
        self._refresh_crop_display()
        self.status_var.set(f"Detected {len(self.current_detections)} objects, {len(self.lcd_crops)} LCD region(s).")

    def _refresh_main_display(self):
        img = self.current_annotated if self.current_annotated is not None else self.current_image_bgr
        if img is None:
            return
        cw = self.canvas_main.winfo_width()
        ch = self.canvas_main.winfo_height()
        if cw <= 1 or ch <= 1:
            return
        fitted = fit_image_to_label(img, cw, ch)
        if fitted is None:
            return
        self.photo_main = cv2_to_photoimage(fitted)
        self.canvas_main.delete("all")
        self.canvas_main.create_image(cw // 2, ch // 2, image=self.photo_main, tags="img")

    def _refresh_crop_display(self):
        idx = 0  # only one LCD
        if idx != self.current_crop_index:
            crop = self.lcd_crops[idx] if self.lcd_crops and idx < len(self.lcd_crops) else None
            init_angle = _default_rotation_for_crop(crop) if crop is not None else 180.0
            self.rotate_var.set(float(init_angle))
            self.rotate_label.config(text=f"{int(round(init_angle))}")
        self.current_crop_index = idx
        crop = self.lcd_crops[idx] if self.lcd_crops else None
        if crop is not None:
            display = self._get_processed_crop(crop)
            if display is not None and len(display.shape) == 2:
                display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
            to_fit = display if (display is not None and getattr(display, "size", 0) > 0) else crop
            fitted = fit_image_to_label(to_fit, 320, 240)
            self.photo_crop = cv2_to_photoimage(fitted)
        else:
            self.photo_crop = None
        self.canvas_crop.delete("all")
        if self.photo_crop:
            cw = max(1, self.canvas_crop.winfo_width() or 320)
            ch = max(1, self.canvas_crop.winfo_height() or 240)
            self.canvas_crop.create_image(cw // 2, ch // 2, image=self.photo_crop, tags="crop")
        # Auto-run OCR when warped image changes (debounced)
        if self.lcd_crops:
            if self._ocr_after_id is not None:
                self.root.after_cancel(self._ocr_after_id)
            self._ocr_after_id = self.root.after(300, self._do_auto_ocr)

    def _do_auto_ocr(self):
        """Called after debounce to run OCR without requiring button press."""
        self._ocr_after_id = None
        self._run_ocr_model()

    def _update_ocr_verdict(self, display_text):
        """Set OK (green) if parsed Nm in [60, 80], else NG (red). Use — if not parseable."""
        value = parse_nm_from_ocr_text(display_text)
        if value is None or display_text in ("No LCD crop available.", "") or "OCR model not found" in (display_text or ""):
            self.ocr_verdict_label.config(text="—", bg="#333", fg="#aaa")
            return
        if is_nm_in_range(value, 60.0, 80.0):
            self.ocr_verdict_label.config(text="OK", bg="#0a5f0a", fg="white")
        else:
            self.ocr_verdict_label.config(text="NG", bg="#8b0000", fg="white")

    def _run_ocr_model(self):
        """Run OCR model on LCD crop (same angle); draw bboxes in OCR result canvas and show text."""
        if not self.lcd_crops:
            self.ocr_text.config(state=tk.NORMAL)
            self.ocr_text.delete("1.0", tk.END)
            self.ocr_text.insert(tk.END, "No LCD crop available.")
            self.ocr_text.config(state=tk.DISABLED)
            self._update_ocr_verdict("No LCD crop available.")
            return
        if self.ocr_model is None:
            self.ocr_model = load_ocr_model()
            if self.ocr_model is None:
                self.ocr_text.config(state=tk.NORMAL)
                self.ocr_text.delete("1.0", tk.END)
                self.ocr_text.insert(tk.END, f"OCR model not found.\nExpected: {OCR_MODEL_WEIGHTS}")
                self.ocr_text.config(state=tk.DISABLED)
                self._update_ocr_verdict("OCR model not found.")
                return
        idx = self.current_crop_index
        crop = self.lcd_crops[idx]
        to_ocr = self._get_processed_crop(crop)
        if to_ocr is None or getattr(to_ocr, "size", 0) == 0:
            to_ocr = crop
        # Add black border so edge letters are not cut off (avoid 'or' with numpy array)
        to_ocr_padded = _ocr_padded_or_original(to_ocr)
        # One run to check if "nm" is in front (flip 180 and retry)
        _, raw_check = run_ocr_model_annotated(self.ocr_model, to_ocr_padded)
        flipped_180 = False
        if raw_check and raw_check.strip().lower().startswith("nm"):
            to_ocr_180 = cv2.rotate(to_ocr, cv2.ROTATE_180)
            to_ocr_180_padded = _ocr_padded_or_original(to_ocr_180)
            display_text, vis, n_tries, filter_idx, raw_str = _ocr_format_with_retries(self.ocr_model, to_ocr_180_padded)
            a = (self.rotate_var.get() + 180) % 360
            if a > 180:
                a -= 360
            self.rotate_var.set(float(a))
            self.rotate_label.config(text=f"{int(round(a))}")
            self._refresh_crop_display()
            flipped_180 = True
        else:
            display_text, vis, n_tries, filter_idx, raw_str = _ocr_format_with_retries(self.ocr_model, to_ocr_padded)
        self._refresh_ocr_result_canvas(vis)
        clean_text = _ocr_clean_display(display_text)
        print(f"Recognize number (raw): {raw_str}")
        self.ocr_text.config(state=tk.NORMAL)
        self.ocr_text.delete("1.0", tk.END)
        self.ocr_text.insert(tk.END, clean_text)
        self.ocr_text.config(state=tk.DISABLED)
        self._update_ocr_verdict(clean_text)
        times_str = "1 time" if n_tries == 1 else f"{n_tries} times"
        filter_name = OCR_FILTER_NAMES[filter_idx] if 0 <= filter_idx < len(OCR_FILTER_NAMES) else "—"
        status = f"OCR done. Run {times_str}. Using {filter_name}."
        if flipped_180:
            status += " (flipped 180°: nm was in front)"
        self.status_var.set(status)

    def _export_all_to_ocr_model(self):
        """Run detection on all images in Raw Image folder; save each LCD crop (as on right panel) to OCR_Model folder."""
        if self.model is None:
            messagebox.showerror("Export", "Model not loaded.")
            return
        raw_images = get_raw_images()
        if not raw_images:
            messagebox.showinfo(
                "Export",
                f"No images found in Raw Image folder.\nAdd images to:\n{RAW_IMAGE_DIR}",
            )
            return
        OCR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        transform_name = self.transform_var.get()
        total_saved = 0
        for path in raw_images:
            img = cv2.imread(str(path))
            if img is None:
                continue
            _, detections = run_inference(self.model, img)
            crops, _ = crop_lcd_regions(img, detections, LCD_CLASS_ID)
            for i, crop in enumerate(crops):
                angle_deg = _default_rotation_for_crop(crop)
                processed = process_crop_for_export(crop, angle_deg, transform_name)
                if processed is None or getattr(processed, "size", 0) == 0:
                    continue
                out_name = f"{path.stem}_lcd{i}.png"
                out_path = OCR_MODEL_DIR / out_name
                cv2.imwrite(str(out_path), processed)
                total_saved += 1
        self.status_var.set(f"Exported {total_saved} crop(s) from {len(raw_images)} image(s) to {OCR_MODEL_DIR}.")
        messagebox.showinfo("Export", f"Saved {total_saved} LCD crop image(s) to:\n{OCR_MODEL_DIR}")


def main():
    root = None
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()
    root.geometry("1200x800")
    root.minsize(900, 600)
    app = IPVApp(root)
    if hasattr(root, "drop_target_register"):
        try:
            root.drop_target_register(DND_FILES)
            root.dnd_bind("<<Drop>>", lambda e: app._load_and_run((e.data or "").strip("{}")))
        except Exception:
            pass
    app.root.mainloop()


if __name__ == "__main__":
    main()
