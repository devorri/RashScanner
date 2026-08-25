"""
train_yolo11.py - Train YOLOv11 Nano Skin Lesion Detection Model & Export to Edge TFLite
Dataset: Roboflow YOLOv11 (12 Classes)
Output: Edge-optimized PyTorch (.pt), ONNX (.onnx), and TensorFlow Lite (.tflite) models
"""

import os
import sys
import yaml
from pathlib import Path

def train():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] Ultralytics is not installed. Please install it using: pip install ultralytics")
        sys.exit(1)

    workspace_dir = Path(__file__).resolve().parent
    dataset_dir = workspace_dir / "Skin Disease Detection.v1i.yolov11"
    yaml_path = dataset_dir / "data.yaml"

    if not yaml_path.exists():
        print(f"[Error] data.yaml not found at: {yaml_path}")
        sys.exit(1)

    # Read classes from data.yaml and update labels.txt
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    
    classes = data_cfg.get("names", [])
    labels_file = workspace_dir / "labels.txt"
    with open(labels_file, "w", encoding="utf-8") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"[+] Loaded {len(classes)} classes from data.yaml and saved to labels.txt:")
    for i, name in enumerate(classes):
        print(f"    {i}: {name}")

    # Fix relative paths in data.yaml to absolute paths
    data_cfg["path"] = str(dataset_dir)
    data_cfg["train"] = str(dataset_dir / "train" / "images")
    data_cfg["val"] = str(dataset_dir / "valid" / "images")
    data_cfg["test"] = str(dataset_dir / "test" / "images")

    fixed_yaml = dataset_dir / "data_fixed.yaml"
    with open(fixed_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_cfg, f)

    print("\n" + "="*60)
    print("🚀 STARTING YOLOV11 NANO TRAINING")
    print("="*60)

    # Load pretrained YOLOv11 Nano
    model = YOLO("yolo11n.pt")

    # Train model
    results = model.train(
        data=str(fixed_yaml),
        epochs=50,
        imgsz=512,
        batch=16,
        name="rash_yolo11_run",
        project=str(workspace_dir / "runs"),
        exist_ok=True,
        device="",  # Auto (GPU if available, CPU otherwise)
        verbose=True
    )

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE! EXPORTING TO TFLITE & ONNX")
    print("="*60)

    # Export best model to TFLite and ONNX for edge deployment
    best_pt = workspace_dir / "runs" / "rash_yolo11_run" / "weights" / "best.pt"
    if best_pt.exists():
        trained_model = YOLO(str(best_pt))
        
        # Export to ONNX
        print("[+] Exporting to ONNX...")
        trained_model.export(format="onnx", imgsz=512)
        
        # Export to TFLite
        print("[+] Exporting to TFLite (Float16)...")
        tflite_path = trained_model.export(format="tflite", imgsz=512, half=True)
        print(f"\n🎉 Edge TFLite model exported successfully: {tflite_path}")
    else:
        print("[!] Warning: best.pt weights not found.")

if __name__ == "__main__":
    train()
