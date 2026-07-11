from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.yaml")

# Load YOLOv8 Nano model
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data=DATA_PATH,
    epochs=10,
    imgsz=640,
    batch=16,
    project="runs",
    name="sign_language",
    device="cpu"
)

print("===================================")
print("Training Completed Successfully!")
print("===================================")
print("Best model saved at:")
print("runs/detect/sign_language/weights/best.pt")