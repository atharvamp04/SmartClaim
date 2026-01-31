# train_yolo_damage.py
from ultralytics import YOLO
import torch
import os

def train_yolo_damage():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    dataset_yaml = "yolo_dataset_damages/dataset.yaml"

    if not os.path.exists(dataset_yaml):
        print("❌ ERROR: Damage dataset YAML not found.")
        print("Run prepare_dataset.py first.")
        return

    print("📦 Loading YOLOv8 nano model for DAMAGE DETECTION...")
    model = YOLO("yolov8n.pt")

    print("🚀 Training started...")
    results = model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=8 if device == "cuda" else 4,
        patience=20,
        device=device,
        name="car_damage_model",
        project="runs/train",
        verbose=True
    )

    best_model = "runs/train/car_damage_model/weights/best.pt"

    if os.path.exists(best_model):
        os.makedirs("models", exist_ok=True)
        os.system(f"copy {best_model} models\\yolov8_car_damage.pt")
        print("✅ DAMAGE model saved to: models/yolov8_car_damage.pt")
    else:
        print("⚠️ Best model not found. Check training run folder.")

if __name__ == "__main__":
    train_yolo_damage()
