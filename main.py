from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo26m-obb.pt")
    model.train(
        data="dataset.yaml", 
        epochs=100, 
        imgsz=640, 
        batch=8, 
        workers=0,  # Set to 0 to avoid multiprocessing issues on Windows
        device=0
    )