"""
Core OCR inference pipeline — stateless, no Tkinter.

Ported from application/ipv.py. All functions are pure (image in → data out).
Entry point: run_full_pipeline(det_model, ocr_model, image_bgr) -> PipelineResult
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

# ── Constants (must match application/ipv.py) ─────────────────────────────────

CLASS_NAMES = ["lcd_display", "torque_wrench"]
LCD_CLASS_ID = 0
OCR_CLASS_NAMES = [".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "nm", "mp"]
OCR_FILTER_NAMES = [
    "Original",
    "Grayscale",
    "Binary (adaptive)",
    "Grayscale + Resize 2x",
    "Border (morph gradient)",
    "Invert",
    "Contrast",
]
OCR_PADDING = 24
OCR_FORMAT_RE = re.compile(r"^\d{2}\.\d{1}(?:nm|mp)$")
NM_RANGE_LOW = 60.0
NM_RANGE_HIGH = 80.0


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    reading: str           # e.g. "72.3nm" or "(can not read image)"
    verdict: str           # "OK" | "NG" | "unknown"
    value_nm: float | None
    n_tries: int
    filter_used: str
    annotated_image: str   # data:image/jpeg;base64,...
    lcd_crop: str | None   # data:image/jpeg;base64,... or None


# ── Image encoding ────────────────────────────────────────────────────────────

def _encode(bgr: np.ndarray | None) -> str | None:
    if bgr is None or bgr.size == 0:
        return None
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# ── OCR format helpers ────────────────────────────────────────────────────────

def _ocr_validate_format(text: str) -> bool:
    return bool(text and OCR_FORMAT_RE.match(text.strip().lower()))


def _ocr_fix_format(raw: str) -> tuple[str | None, bool]:
    if not raw:
        return None, False
    digits_only = "".join(c for c in raw if c.isdigit())
    s = "".join(c for c in raw if c.isdigit() or c == ".")
    if _ocr_validate_format(raw.strip()):
        return raw.strip().lower().replace("mp", "nm"), False
    if len(s) == 4 and s[2] == "." and s[:2].isdigit() and s[3].isdigit():
        return s + "nm", True
    if len(digits_only) == 3:
        return f"{digits_only[0]}{digits_only[1]}.{digits_only[2]}nm", True
    if len(digits_only) >= 3:
        return f"{digits_only[0]}{digits_only[1]}.{digits_only[-1]}nm", True
    if "." in s:
        parts = s.split(".")
        if len(parts) == 2 and len(parts[0]) >= 2 and parts[1]:
            a = parts[0][-2:]
            b = parts[1][0]
            return f"{a}.{b}nm", True
    return None, False


def _ocr_clean_display(text: str) -> str:
    if not text:
        return text
    return text.split(" (")[0].strip()


# ── OCR preprocessing filters ─────────────────────────────────────────────────

def _ocr_bgr_to_binary_adaptive(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(g, (3, 3), 0)
    out = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3.5
    )
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _ocr_bgr_to_border(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    out = cv2.morphologyEx(g, cv2.MORPH_GRADIENT, kernel)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _ocr_invert(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(cv2.bitwise_not(g), cv2.COLOR_GRAY2BGR)


def _ocr_contrast(bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    out = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _apply_filter(bgr: np.ndarray, index: int) -> np.ndarray:
    filters = [
        lambda im: im,
        lambda im: cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
        lambda im: _ocr_bgr_to_binary_adaptive(im),
        lambda im: cv2.cvtColor(
            cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=2, fy=2,
                       interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR
        ),
        lambda im: _ocr_bgr_to_border(im),
        lambda im: _ocr_invert(im),
        lambda im: _ocr_contrast(im),
    ]
    out = filters[index % len(filters)](bgr)
    return out if out is not None and out.size > 0 else bgr


# ── OCR model runner ──────────────────────────────────────────────────────────

def _dedupe_items(
    items: list[tuple], img_width: int, min_gap_frac: float = 0.03
) -> list[tuple]:
    min_gap = max(4, img_width * min_gap_frac)
    dedup: list[tuple] = []
    for t in items:
        x, conf = t[0], t[2] if len(t) >= 3 else 1.0
        if dedup and abs(x - dedup[-1][0]) < min_gap:
            if len(dedup[-1]) >= 3 and conf > dedup[-1][2]:
                dedup[-1] = t
        else:
            dedup.append(t)
    return dedup


def _add_padding(bgr: np.ndarray, padding: int = OCR_PADDING) -> np.ndarray:
    if bgr is None or bgr.size == 0:
        return bgr
    img = bgr if len(bgr.shape) == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    out = np.zeros((h + 2 * padding, w + 2 * padding, 3), dtype=img.dtype)
    out[padding: padding + h, padding: padding + w] = img
    return out


def _run_ocr_annotated(ocr_model: Any, bgr: np.ndarray) -> tuple[np.ndarray, str]:
    """Run OCR YOLO on image; return (annotated image, decoded string)."""
    if ocr_model is None or bgr is None or bgr.size == 0:
        return bgr, ""
    img = bgr if len(bgr.shape) == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    vis = img.copy()
    try:
        results = ocr_model(img, verbose=False)
        r = results[0] if results else None
        if r is None or not hasattr(r, "obb") or r.obb is None or len(r.obb) == 0:
            return vis, ""
        items = []
        for i in range(len(r.obb)):
            cls_id = int(r.obb.cls[i].item())
            conf = float(r.obb.conf[i].item())
            pts = (
                r.obb.xyxyxyxy[i].cpu().numpy()
                if hasattr(r.obb.xyxyxyxy[i], "cpu")
                else r.obb.xyxyxyxy[i]
            )
            corners = np.array(pts, dtype=np.int32).reshape(4, 2)
            center_x = float(corners[:, 0].mean())
            items.append((center_x, cls_id, conf, corners))
        items.sort(key=lambda x: x[0])
        items = _dedupe_items(items, img.shape[1])
        text = "".join(
            OCR_CLASS_NAMES[cid] if 0 <= cid < len(OCR_CLASS_NAMES) else "?"
            for _, cid, _, _ in items
        )
        for _, cls_id, _, corners in items:
            label = OCR_CLASS_NAMES[cls_id] if 0 <= cls_id < len(OCR_CLASS_NAMES) else "?"
            cv2.polylines(vis, [corners], isClosed=True, color=(0, 255, 0), thickness=2)
            bottom_y = int(corners[:, 1].max())
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cx = int(corners[:, 0].mean())
            cv2.putText(
                vis, label,
                (cx - tw // 2, bottom_y + th + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )
        return vis, text
    except Exception as exc:
        return vis, f"OCR error: {exc}"


def _ocr_with_retries(
    ocr_model: Any, bgr: np.ndarray, max_tries: int = 7
) -> tuple[str, np.ndarray | None, int, int, str]:
    """Try up to max_tries filters; return (display_text, vis, n_tries, filter_idx, raw)."""
    best_fixed = best_vis = None
    best_was_fixed = False
    best_attempt = 0
    best_raw = ""
    last_vis = last_raw = None

    for attempt in range(max_tries):
        filtered = _apply_filter(bgr, attempt)
        vis, raw = _run_ocr_annotated(ocr_model, filtered)
        if vis is not None:
            last_vis = vis
        last_raw = raw or ""
        fixed, was_fixed = _ocr_fix_format(last_raw)
        if fixed and _ocr_validate_format(fixed):
            display = f"{fixed} (fixed)" if was_fixed else fixed
            return display, last_vis, attempt + 1, attempt, last_raw
        if fixed:
            best_fixed, best_was_fixed = fixed, was_fixed
            best_vis = last_vis
            best_attempt = attempt
            best_raw = last_raw

    if best_fixed:
        label = f"{best_fixed} (fixed) ({max_tries} tried)"
        return label, best_vis or last_vis, max_tries, best_attempt, best_raw

    if not last_raw or not any(c.isdigit() for c in last_raw):
        return "(can not read image)", last_vis, max_tries, -1, last_raw or ""

    return f"{last_raw.strip()} ({max_tries} tried)", last_vis, max_tries, -1, last_raw


# ── Detection helpers ─────────────────────────────────────────────────────────

def _is_obb_detection(det: tuple) -> bool:
    if len(det) != 4:
        return False
    return np.asarray(det[2]).shape == (4, 2)


def _order_quad_tl_tr_br_bl(corners: np.ndarray) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    ordered = pts[np.argsort(angles)]
    tl_pos = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    rotated = np.roll(ordered, -tl_pos, axis=0)
    if rotated[1][0] <= rotated[0][0]:
        rotated[1:4] = rotated[3:0:-1]
    return rotated.astype(np.float32)


def _default_rotation(crop: np.ndarray) -> float:
    if crop is None or crop.size == 0:
        return 180.0
    h, w = crop.shape[:2]
    return 90.0 if h > w else 180.0


def _apply_rotation(bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.01:
        return bgr
    h, w = bgr.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    M[0, 2] += nw / 2.0 - cx
    M[1, 2] += nh / 2.0 - cy
    return cv2.warpAffine(bgr, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT)


def _run_detection(det_model: Any, bgr: np.ndarray) -> tuple[np.ndarray, list]:
    results = det_model(bgr, verbose=False)
    detections = []
    out = bgr.copy()
    r = results[0] if results else None
    if r is not None and hasattr(r, "obb") and r.obb is not None and len(r.obb) > 0:
        for i in range(len(r.obb)):
            cls_id = int(r.obb.cls[i].item())
            conf = float(r.obb.conf[i].item())
            pts = (
                r.obb.xyxyxyxy[i].cpu().numpy()
                if hasattr(r.obb.xyxyxyxy[i], "cpu")
                else r.obb.xyxyxyxy[i]
            )
            corners = np.array(pts, dtype=np.int32).reshape(4, 2)
            xywhr = (
                r.obb.xywhr[i].cpu().numpy()
                if hasattr(r.obb.xywhr[i], "cpu")
                else r.obb.xywhr[i]
            )
            r_val = float(xywhr[4])
            angle_deg = np.degrees(r_val) if abs(r_val) <= np.pi + 0.1 else r_val
            detections.append((cls_id, conf, corners, angle_deg))
            color = (0, 255, 0) if cls_id == LCD_CLASS_ID else (255, 165, 0)
            cv2.polylines(out, [corners], isClosed=True, color=color, thickness=2)
            label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
            cv2.putText(out, label,
                        (int(corners[0][0]), int(corners[0][1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    else:
        if r is not None and r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((cls_id, x1, y1, x2, y2, conf))
                color = (0, 255, 0) if cls_id == LCD_CLASS_ID else (255, 165, 0)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"
                cv2.putText(out, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return out, detections


def _crop_lcd_regions(
    bgr: np.ndarray, detections: list
) -> tuple[list[np.ndarray], list[float]]:
    h, w = bgr.shape[:2]
    crops, angles = [], []
    for det in detections:
        if _is_obb_detection(det):
            cid, conf, corners, angle_deg = det
            if cid != LCD_CLASS_ID:
                continue
            src = _order_quad_tl_tr_br_bl(corners)
            w1 = np.linalg.norm(src[1] - src[0])
            w2 = np.linalg.norm(src[2] - src[3])
            h1 = np.linalg.norm(src[3] - src[0])
            h2 = np.linalg.norm(src[2] - src[1])
            dw = int(max(10, (w1 + w2) / 2))
            dh = int(max(10, (h1 + h2) / 2))
            dst = np.array([[0, 0], [dw, 0], [dw, dh], [0, dh]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)
            crop = cv2.warpPerspective(bgr, M, (dw, dh), flags=cv2.INTER_LINEAR)
            crops.append(crop)
            angles.append(angle_deg)
        else:
            cid, x1, y1, x2, y2, _ = det
            if cid != LCD_CLASS_ID:
                continue
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                crops.append(bgr[y1:y2, x1:x2].copy())
                angles.append(0.0)
    return crops, angles


# ── Public entry point ────────────────────────────────────────────────────────

def parse_nm(display_text: str) -> float | None:
    if not display_text:
        return None
    m = re.search(r"(\d+\.?\d*)", display_text.strip())
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def run_full_pipeline(
    det_model: Any,
    ocr_model: Any,
    image_bgr: np.ndarray,
) -> PipelineResult:
    """
    Full pipeline: detect LCD → crop → rotate → OCR → validate.
    Returns a PipelineResult ready to be serialised as JSON.
    """
    # 1. Object detection
    annotated, detections = _run_detection(det_model, image_bgr)

    # 2. Crop LCD regions
    crops, angles = _crop_lcd_regions(image_bgr, detections)

    if not crops:
        return PipelineResult(
            reading="(no LCD detected)",
            verdict="unknown",
            value_nm=None,
            n_tries=0,
            filter_used="—",
            annotated_image=_encode(annotated),
            lcd_crop=None,
        )

    # 3. Use the first (most confident) LCD crop
    crop = crops[0]
    init_angle = _default_rotation(crop)
    rotated = _apply_rotation(crop, init_angle)

    # 4. Run OCR with 7-filter retry strategy
    if ocr_model is not None:
        padded = _add_padding(rotated)

        # Auto-flip 180° if "nm" appears at the front
        _, raw_check = _run_ocr_annotated(ocr_model, padded)
        if raw_check and raw_check.strip().lower().startswith("nm"):
            rotated = cv2.rotate(rotated, cv2.ROTATE_180)
            padded = _add_padding(rotated)

        display, vis, n_tries, filter_idx, _ = _ocr_with_retries(ocr_model, padded)
        clean = _ocr_clean_display(display)
        filter_name = (
            OCR_FILTER_NAMES[filter_idx]
            if 0 <= filter_idx < len(OCR_FILTER_NAMES)
            else "—"
        )
    else:
        clean = "(OCR model not loaded)"
        vis = rotated
        n_tries = 0
        filter_name = "—"

    # 5. Validate range
    value_nm = parse_nm(clean)
    if value_nm is None:
        verdict = "unknown"
    elif NM_RANGE_LOW <= value_nm <= NM_RANGE_HIGH:
        verdict = "OK"
    else:
        verdict = "NG"

    return PipelineResult(
        reading=clean,
        verdict=verdict,
        value_nm=value_nm,
        n_tries=n_tries,
        filter_used=filter_name,
        annotated_image=_encode(annotated),
        lcd_crop=_encode(vis if vis is not None else rotated),
    )
