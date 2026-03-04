from ultralytics import YOLO
import os
import cv2
import shutil
import numpy as np

# Use Tesseract with ssd (seven-segment) language, 2 digits + 1 decimal (e.g. 82.1)
USE_TESSERACT_SSD = True   # set False to use EasyOCR instead
TESSERACT_DIGITS_BEFORE = 2
TESSERACT_DIGITS_AFTER = 1

# Clear output directory
output_dir = "val_results"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

# Load model
model = YOLO("runs/detect/train7/weights/best.pt")
if not USE_TESSERACT_SSD:
    import easyocr
    print("Loading EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=True)
else:
    import tesseract_ssd
    reader = None
    print("Using Tesseract OCR (lang=ssd, 2 digits + 1 decimal)")

val_images_dir = "train/images"
image_files = [f for f in os.listdir(val_images_dir) 
               if f.endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(image_files)} images")
print("=" * 70)

def preprocess_display(display_img):
    """Enhance display image for OCR - NO INVERSION"""
    gray = cv2.cvtColor(display_img, cv2.COLOR_BGR2GRAY)
    
    # Upscale
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    
    # Denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean noise
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh

for img_file in image_files:
    img_path = os.path.join(val_images_dir, img_file)
    print(f"Processing: {img_file}")
    
    results = model(img_path)
    img = cv2.imread(img_path)
    
    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            color = (0, 255, 0) if class_name == "lcd_display" else (255, 0, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            if class_name == "lcd_display":
                display = img[y1:y2, x1:x2]
                
                # Preprocess (no inversion)
                processed = preprocess_display(display)
                
                # Save debug image
                debug_path = os.path.join(output_dir, f"debug_{img_file}")
                cv2.imwrite(debug_path, processed)
                
                try:
                    if USE_TESSERACT_SSD:
                        best_reading = tesseract_ssd.read_7segment_tesseract(
                            processed,
                            digits_before=TESSERACT_DIGITS_BEFORE,
                            digits_after=TESSERACT_DIGITS_AFTER,
                        )
                        best_conf = 1.0
                        if best_reading and not best_reading.startswith("("):
                            # Draw on image
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.2
                            thickness = 3
                            (text_width, text_height), _ = cv2.getTextSize(
                                best_reading, font, font_scale, thickness
                            )
                            cv2.rectangle(img,
                                         (x1, y1 - text_height - 20),
                                         (x1 + text_width + 10, y1),
                                         (0, 255, 0), -1)
                            cv2.putText(img, best_reading, (x1 + 5, y1 - 10),
                                       font, font_scale, (0, 0, 0), thickness)
                            print(f"  ✓ {class_name} (conf: {conf:.1%})")
                            print(f"    📊 Reading: {best_reading} (Tesseract ssd)")
                        else:
                            print(f"  ✓ {class_name} (conf: {conf:.1%})")
                            print(f"    ❌ No reading")
                            cv2.putText(img, "No reading", (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    else:
                        ocr_results = reader.readtext(
                            processed,
                            allowlist='0123456789.-',
                            detail=1,
                            paragraph=False
                        )
                        if ocr_results:
                            readings = [r[1] for r in ocr_results]
                            confidences = [r[2] for r in ocr_results]
                            best_reading = max(readings, key=len) if readings else ""
                            best_conf = confidences[readings.index(best_reading)] if readings else 0
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 1.2
                            thickness = 3
                            (text_width, text_height), _ = cv2.getTextSize(
                                best_reading, font, font_scale, thickness
                            )
                            cv2.rectangle(img,
                                         (x1, y1 - text_height - 20),
                                         (x1 + text_width + 10, y1),
                                         (0, 255, 0), -1)
                            cv2.putText(img, best_reading, (x1 + 5, y1 - 10),
                                       font, font_scale, (0, 0, 0), thickness)
                            print(f"  ✓ {class_name} (conf: {conf:.1%})")
                            print(f"    📊 Reading: {best_reading} (OCR: {best_conf:.1%})")
                            if len(readings) > 1:
                                print(f"    All detected: {readings}")
                        else:
                            print(f"  ✓ {class_name} (conf: {conf:.1%})")
                            print(f"    ❌ No reading")
                            cv2.putText(img, "No reading", (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                except Exception as e:
                    print(f"  ⚠️ OCR error: {e}")
            else:
                label = f"{class_name} {conf:.0%}"
                cv2.putText(img, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                print(f"  ✓ {class_name} (conf: {conf:.1%})")
    else:
        print("  ❌ No detections")
        cv2.putText(img, "No detections", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    output_path = os.path.join(output_dir, f"result_{img_file}")
    cv2.imwrite(output_path, img)
    print(f"  💾 Saved\n")

print("=" * 70)
print(f"✅ Check {output_dir}/ for results")
print(f"Check debug_*.jpg to see what OCR sees")
print("=" * 70)