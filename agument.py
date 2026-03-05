import cv2
import albumentations as A
import os

def augment_training_data():
    # Fixed SafeRotate and updated to modern Albumentations style
    transform = A.Compose([
        # fill_value is the correct argument for SafeRotate
        A.SafeRotate(limit=20, p=0.8, border_mode=cv2.BORDER_CONSTANT, fill_value=0),
        
        # Using Affine as suggested by the warning for better compatibility
        A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, 
                 scale=(0.9, 1.1), rotate=0, p=0.5),
        
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.Blur(blur_limit=3, p=0.2), 
        
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    input_dir = "trainobb(base)"
    output_dir = "trainobb"
    
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)
    
    image_files = [f for f in os.listdir(f"{input_dir}/images") if f.endswith(('.jpg', '.png'))]
    
    print(f"Starting augmentation on {len(image_files)} images...")

    for img_file in image_files:
        img_path = os.path.join(input_dir, "images", img_file)
        label_file = img_file.rsplit('.', 1)[0] + ".txt"
        label_path = os.path.join(input_dir, "labels", label_file)
        
        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        bboxes, class_labels = [], []
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            # FIX: Convert to float first, THEN to int to handle '0.0'
            class_labels.append(int(float(parts[0]))) 
            bboxes.append([float(x) for x in parts[1:5]])
        
        # Save Original
        cv2.imwrite(f"{output_dir}/images/{img_file}", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        with open(f"{output_dir}/labels/{label_file}", 'w') as f:
            for cls, bbox in zip(class_labels, bboxes):
                f.write(f"{cls} {' '.join(map(str, bbox))}\n")

        # Generate 3 Augmented versions
        for i in range(3):
            try:
                transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                
                if len(transformed['bboxes']) == len(bboxes):
                    aug_img = transformed['image']
                    aug_bboxes = transformed['bboxes']
                    aug_labels = transformed['class_labels']
                    
                    base_name = img_file.rsplit('.', 1)[0]
                    aug_name = f"{base_name}_aug_{i}"
                    
                    cv2.imwrite(f"{output_dir}/images/{aug_name}.jpg", cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                    with open(f"{output_dir}/labels/{aug_name}.txt", 'w') as f:
                        for cls, bbox in zip(aug_labels, aug_bboxes):
                            f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
            except Exception as e:
                print(f"Skipping index {i} for {img_file}: {e}")

    print(f"\n✅ Done! Check your {output_dir} folder.")

if __name__ == '__main__':
    augment_training_data()