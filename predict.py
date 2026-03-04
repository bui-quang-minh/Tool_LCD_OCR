from ultralytics import YOLO
import os
import cv2

# Load trained model
model = YOLO("runs/detect/train7/weights/best.pt")

# Input and output directories
val_images_dir = "train/images"
output_dir = "val_results"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Get all image files
image_files = [f for f in os.listdir(val_images_dir) 
               if f.endswith(('.jpg', '.jpeg', '.png'))]

print(f"Found {len(image_files)} images in {val_images_dir}")
print("-" * 60)

# Process each image
for img_file in image_files:
    img_path = os.path.join(val_images_dir, img_file)
    
    print(f"\nProcessing: {img_file}")
    
    # Run detection
    results = model(img_path)
    
    # Save result with bounding boxes
    output_path = os.path.join(output_dir, f"result_{img_file}")
    results[0].save(output_path)
    
    # Print detections
    boxes = results[0].boxes
    if len(boxes) > 0:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            
            print(f"  - {class_name}: confidence={conf:.2f}, box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
    else:
        print("  - No detections")
    
    print(f"  Saved to: {output_path}")

print("\n" + "=" * 60)
print(f"All results saved to: {output_dir}/")
print("=" * 60)