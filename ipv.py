"""
IPV - Tool & LCD detection app.
Option 1: Run YOLO on validation set.
Option 2: Load image (file dialog or drag-and-drop).
Shows bounding boxes, LCD crop, transforms for OCR, and Tesseract OCR.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from PIL import Image, ImageTk
import cv2
import numpy as np

# Project paths
PROJECT_ROOT = Path(r"F:\FSB\Sem1\IPV")
MODEL_PATH = PROJECT_ROOT / r"runs\detect\train10\weights\best.pt"
VAL_DIR = PROJECT_ROOT / "val\images"
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Seven-segment display format: 2 digits before decimal, 1 after (e.g. 82.1)
TESSERACT_DIGITS_BEFORE = 2
TESSERACT_DIGITS_AFTER = 1

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


def load_yolo():
    """Load YOLO model."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(MODEL_PATH))
        return model
    except Exception as e:
        messagebox.showerror("Model load error", str(e))
        return None


def run_inference(model, image_bgr):
    """Run YOLO on BGR image; return (annotated BGR image, list of detections)."""
    results = model(image_bgr, verbose=False)
    detections = []  # (class_id, x1, y1, x2, y2, conf)
    out = image_bgr.copy()
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


def crop_lcd_regions(image_bgr, detections, class_id=LCD_CLASS_ID):
    """Return list of BGR crops for given class (default LCD)."""
    h, w = image_bgr.shape[:2]
    crops = []
    for det in detections:
        cid, x1, y1, x2, y2, _ = det
        if cid != class_id:
            continue
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            crops.append(image_bgr[y1:y2, x1:x2].copy())
    return crops


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


def run_tesseract(image_bgr_or_gray, digits_before=None, digits_after=None):
    """Run Tesseract OCR with ssd (seven-segment) language. Format: 2 digits, 1 decimal (e.g. 82.1)."""
    if digits_before is None:
        digits_before = TESSERACT_DIGITS_BEFORE
    if digits_after is None:
        digits_after = TESSERACT_DIGITS_AFTER
    try:
        import tesseract_ssd
        return tesseract_ssd.read_7segment_tesseract(
            image_bgr_or_gray,
            digits_before=digits_before,
            digits_after=digits_after,
        )
    except Exception as e:
        return f"OCR error: {e}"


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
        self.current_image_bgr = None
        self.current_annotated = None
        self.current_detections = []
        self.lcd_crops = []
        self.current_crop_index = 0
        self.val_image_paths = []
        self.val_index = 0
        self.photo_main = None
        self.photo_crop = None
        self.rotate_angle = 0.0
        self.calibration_quad = None  # list of 4 (x,y) in crop coords, or None = use full rect
        self.calibration_dragging = None  # index of handle being dragged
        self.calib_scale = 1.0
        self.calib_offset_x = 0
        self.calib_offset_y = 0
        self._calib_photo = None

        self._build_ui()
        self._load_model()

    def _load_model(self):
        self.model = load_yolo()
        if self.model is not None:
            self.status_var.set("Model loaded.")
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

        # Calibration (drag corners)
        ttk.Label(right, text="Calibration (drag corners):", font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(8, 0))
        self.calib_canvas = tk.Canvas(right, bg="#1a1a1a", highlightthickness=1, highlightbackground="#444", width=300, height=160)
        self.calib_canvas.pack(fill=tk.X, pady=2)
        self.calib_canvas.bind("<Button-1>", self._calib_on_press)
        self.calib_canvas.bind("<B1-Motion>", self._calib_on_drag)
        self.calib_canvas.bind("<ButtonRelease-1>", self._calib_on_release)
        ttk.Button(right, text="Reset calibration", command=self._calib_reset).pack(anchor=tk.W, pady=2)

        ttk.Label(right, text="Transform:", font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(8, 0))
        self.transform_var = tk.StringVar(value="None")
        transforms = ["None", "Grayscale", "Binary (Otsu)", "Binary (adaptive)", "Resize 2x", "Grayscale + Resize 2x"]
        cb = ttk.Combobox(right, textvariable=self.transform_var, values=transforms, state="readonly", width=22)
        cb.pack(fill=tk.X, pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._refresh_crop_display())
        ttk.Button(right, text="Run OCR (Tesseract)", command=self._run_ocr).pack(pady=8)
        self.ocr_text = tk.Text(right, height=6, wrap=tk.WORD, font=("Consolas", 9), state=tk.DISABLED)
        self.ocr_text.pack(fill=tk.X, pady=4)
        ttk.Label(right, text="Crop index (if multiple LCDs):", font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(4, 0))
        self.crop_index_var = tk.StringVar(value="0")
        self.spin_crop = ttk.Spinbox(right, from_=0, to=0, textvariable=self.crop_index_var, width=6)
        self.spin_crop.pack(anchor=tk.W, pady=2)
        self.spin_crop.bind("<Return>", lambda e: self._refresh_crop_display())
        self.spin_crop.bind("<<Increment>>", lambda e: self._refresh_crop_display())
        self.spin_crop.bind("<<Decrement>>", lambda e: self._refresh_crop_display())

        # Status
        ttk.Label(main, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(fill=tk.X, pady=(4, 0))

        # Drag and drop is enabled in main() when tkinterdnd2 is available

    def _get_processed_crop(self, crop):
        """Apply calibration -> rotate -> transform to get final crop for display/OCR."""
        if crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        quad = self.calibration_quad
        if quad is None:
            quad = [(0, 0), (w, 0), (w, h), (0, h)]
        out = apply_perspective(crop, quad)
        if out is None:
            out = crop
        out = apply_rotation(out, self.rotate_var.get())
        if out is None:
            out = crop
        out = apply_transform(out, self.transform_var.get())
        return out if (out is not None and getattr(out, "size", 0) > 0) else crop

    def _on_rotate_change(self, *_):
        self.rotate_label.config(text=f"{int(round(self.rotate_var.get()))}")
        self._refresh_crop_display()

    def _set_rotate(self, angle):
        self.rotate_var.set(float(angle))
        self.rotate_label.config(text=f"{int(angle)}")
        self._refresh_crop_display()

    def _calib_reset(self):
        self.calibration_quad = None
        self._refresh_calib_canvas()
        self._refresh_crop_display()

    def _canvas_to_crop(self, cx, cy):
        """Convert calibration canvas coords to crop coords."""
        x = (cx - self.calib_offset_x) / self.calib_scale
        y = (cy - self.calib_offset_y) / self.calib_scale
        return (x, y)

    def _crop_to_canvas(self, x, y):
        """Convert crop coords to calibration canvas coords."""
        cx = x * self.calib_scale + self.calib_offset_x
        cy = y * self.calib_scale + self.calib_offset_y
        return (cx, cy)

    def _calib_on_press(self, event):
        if not self.lcd_crops:
            return
        crop = self.lcd_crops[self.current_crop_index]
        h, w = crop.shape[:2]
        quad = self.calibration_quad if self.calibration_quad is not None else [(0, 0), (w, 0), (w, h), (0, h)]
        handle_radius = 8
        for i, (px, py) in enumerate(quad):
            cx, cy = self._crop_to_canvas(px, py)
            if abs(event.x - cx) <= handle_radius and abs(event.y - cy) <= handle_radius:
                self.calibration_dragging = i
                return
        self.calibration_dragging = None

    def _calib_on_drag(self, event):
        if self.calibration_dragging is None or not self.lcd_crops:
            return
        crop = self.lcd_crops[self.current_crop_index]
        h, w = crop.shape[:2]
        px, py = self._canvas_to_crop(event.x, event.y)
        px = max(0, min(w, px))
        py = max(0, min(h, py))
        if self.calibration_quad is None:
            self.calibration_quad = [(0, 0), (w, 0), (w, h), (0, h)]
        self.calibration_quad[self.calibration_dragging] = (px, py)
        self._refresh_calib_canvas()
        self._refresh_crop_display()

    def _calib_on_release(self, event):
        self.calibration_dragging = None

    def _refresh_calib_canvas(self):
        """Draw base crop and 4 draggable corner handles on calibration canvas."""
        self.calib_canvas.delete("all")
        if not self.lcd_crops:
            return
        crop = self.lcd_crops[self.current_crop_index]
        h, w = crop.shape[:2]
        cw = self.calib_canvas.winfo_width() or 300
        ch = self.calib_canvas.winfo_height() or 160
        scale = min(cw / w, ch / h, 1.0)
        self.calib_scale = scale
        self.calib_offset_x = cw / 2.0 - (w * scale) / 2.0
        self.calib_offset_y = ch / 2.0 - (h * scale) / 2.0
        fitted = fit_image_to_label(crop, cw, ch)
        if fitted is not None:
            self._calib_photo = cv2_to_photoimage(fitted)
            self.calib_canvas.create_image(cw // 2, ch // 2, image=self._calib_photo, tags="calib_img")
        quad = self.calibration_quad if self.calibration_quad is not None else [(0, 0), (w, 0), (w, h), (0, h)]
        for i, (px, py) in enumerate(quad):
            cx, cy = self._crop_to_canvas(px, py)
            r = 6
            self.calib_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#0af", outline="white", width=2, tags="handle")
        if len(quad) >= 4:
            for i in range(4):
                cx1, cy1 = self._crop_to_canvas(quad[i][0], quad[i][1])
                cx2, cy2 = self._crop_to_canvas(quad[(i + 1) % 4][0], quad[(i + 1) % 4][1])
                self.calib_canvas.create_line(cx1, cy1, cx2, cy2, fill="#0af", width=1, dash=(2, 2))

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
        self.lcd_crops = crop_lcd_regions(self.current_image_bgr, self.current_detections, LCD_CLASS_ID)
        self.current_crop_index = 0
        self.calibration_quad = None
        self.spin_crop.config(to=max(0, len(self.lcd_crops) - 1))
        self.crop_index_var.set("0")
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
        try:
            idx = int(self.crop_index_var.get())
        except ValueError:
            idx = 0
        idx = max(0, min(idx, len(self.lcd_crops) - 1)) if self.lcd_crops else 0
        if idx != self.current_crop_index:
            self.calibration_quad = None  # reset calibration when switching LCD crop
        self.current_crop_index = idx
        self.crop_index_var.set(str(idx))
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
        self._refresh_calib_canvas()

    def _run_ocr(self):
        if not self.lcd_crops:
            self.ocr_text.config(state=tk.NORMAL)
            self.ocr_text.delete("1.0", tk.END)
            self.ocr_text.insert(tk.END, "No LCD crop available.")
            self.ocr_text.config(state=tk.DISABLED)
            return
        idx = self.current_crop_index
        crop = self.lcd_crops[idx]
        to_ocr = self._get_processed_crop(crop)
        if to_ocr is None or getattr(to_ocr, "size", 0) == 0:
            to_ocr = crop
        text = run_tesseract(to_ocr)
        self.ocr_text.config(state=tk.NORMAL)
        self.ocr_text.delete("1.0", tk.END)
        self.ocr_text.insert(tk.END, text)
        self.ocr_text.config(state=tk.DISABLED)
        self.status_var.set("OCR done.")


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
