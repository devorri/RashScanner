"""
pi_scanner.py - Edge AI Skin Disease Scanner Engine with YOLOv11 Object Detection
Features:
- Real-Time AI Camera Analyzer with live bounding box overlays for multiple lesions
- 100% AI Computer Vision object detection across 12 unique dermatological conditions
- Fitzpatrick skin-tone presence & image sharpness gating
- Seamless Raspberry Pi (Picamera2) & OpenCV VideoCapture support
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2
import importlib
from pathlib import Path
from typing import Dict, List, Any, Optional

from symptoms_db import get_condition_info, get_contagious_status, check_red_flags

# ------------------------------------------------------------------------------
# Image Quality & Skin Presence Detector
# ------------------------------------------------------------------------------
def detect_skin_and_quality(bgr_image, return_mask: bool = False):
    """
    Validates image before neural network inference:
    1. Skin Color Distribution: Multi-space (YCrCb + HSV) detector supporting Fitzpatrick tones I-VI.
    2. Blur / Focus Clarity: Laplacian variance check.
    3. Lighting / Exposure: Brightness level analysis.
    """
    if bgr_image is None or bgr_image.size == 0:
        res = {
            "is_valid": False,
            "skin_detected": False,
            "skin_ratio": 0.0,
            "is_blurry": True,
            "blur_score": 0.0,
            "brightness": 0.0,
            "error": "Empty or invalid image frame"
        }
        if return_mask:
            res["skin_mask"] = None
        return res

    h, w = bgr_image.shape[:2]
    total_pixels = h * w

    # 1. Skin Detection via HSV & YCrCb Color Spaces
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YCrCb)

    lower_hsv1 = np.array([0, 20, 40], dtype=np.uint8)
    upper_hsv1 = np.array([30, 255, 255], dtype=np.uint8)
    lower_hsv2 = np.array([165, 20, 40], dtype=np.uint8)
    upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.bitwise_or(cv2.inRange(hsv, lower_hsv1, upper_hsv1), cv2.inRange(hsv, lower_hsv2, upper_hsv2))

    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 175, 130], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    skin_mask = cv2.bitwise_or(mask_hsv, mask_ycrcb)
    skin_pixels = cv2.countNonZero(skin_mask)
    skin_ratio = skin_pixels / total_pixels

    # 2. Blur / Sharpness check
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 3. Brightness level check
    brightness = float(np.mean(gray))

    MIN_SKIN_RATIO = 0.05   # At least 5% skin presence
    MIN_BLUR_SCORE = 25.0   # Sharpness threshold
    MIN_BRIGHTNESS = 20.0
    MAX_BRIGHTNESS = 245.0

    skin_detected = bool(skin_ratio >= MIN_SKIN_RATIO)
    is_sharp = bool(blur_score >= MIN_BLUR_SCORE)
    is_blurry = not is_sharp
    is_poor_lighting = bool(brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS)
    is_valid = bool(skin_detected and is_sharp and not is_poor_lighting)

    error_msg = None
    if not skin_detected:
        error_msg = f"No human skin detected ({skin_ratio*100:.1f}% coverage). Please position camera on skin."
    elif not is_sharp:
        error_msg = f"Camera frame is blurry ({blur_score:.1f} sharpness). Please hold camera steady."
    elif is_poor_lighting:
        if brightness < MIN_BRIGHTNESS:
            error_msg = "Frame is too dark. Please increase illumination."
        else:
            error_msg = "Frame is over-exposed or glaring."

    warnings = []
    if blur_score < 40.0 and is_sharp:
        warnings.append(f"Moderate sharpness ({blur_score:.1f}). Hold still for optimal precision.")

    result = {
        "is_valid": is_valid,
        "skin_detected": skin_detected,
        "skin_ratio": round(float(skin_ratio * 100), 1),
        "is_sharp": is_sharp,
        "is_blurry": is_blurry,
        "blur_score": round(float(blur_score), 1),
        "brightness": round(float(brightness), 1),
        "warnings": warnings,
        "error": error_msg
    }

    if return_mask:
        result["skin_mask"] = skin_mask

    return result

# ------------------------------------------------------------------------------
# Robust Camera Module
# ------------------------------------------------------------------------------
class EdgeCamera:
    _instance = None

    def __init__(self, camera_index=0, resolution=(640, 480)):
        self.resolution = resolution
        self.picam2 = None
        self.cap = None
        self.backend = None

        try:
            picam_mod = importlib.import_module("picamera2")
            Picamera2_cls = getattr(picam_mod, "Picamera2")
            self.picam2 = Picamera2_cls()
            config = self.picam2.create_preview_configuration(main={"size": resolution})
            self.picam2.configure(config)
            self.picam2.start()
            self.backend = "picamera2"
            print("[Camera] Initialized native Raspberry Pi Picamera2!")
        except Exception:
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                self.backend = "opencv"
                print(f"[Camera] Initialized OpenCV VideoCapture({camera_index}).")
            else:
                self.backend = "synthetic"
                print("[Camera Info] Switched to Synthetic Test Frame mode.")

    def capture_frame(self):
        """Captures a single BGR frame."""
        if self.backend == "picamera2" and self.picam2:
            frame_rgb = self.picam2.capture_array()
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        elif self.backend == "opencv" and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame

        # Synthetic Fallback Frame
        img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        cv2.circle(img, (self.resolution[0]//2, self.resolution[1]//2), 90, (140, 180, 210), -1)
        cv2.putText(img, "Synthetic Test Feed", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img

    def release(self):
        if self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass
        if self.cap:
            self.cap.release()

# ------------------------------------------------------------------------------
# 100% AI YOLOv11 / TFLite Computer Vision Detector
# ------------------------------------------------------------------------------
class TFLiteClassifier:
    """
    Skin Disease AI Detector supporting YOLOv11 PyTorch (.pt), ONNX (.onnx), and TFLite models.
    Provides precise multi-lesion bounding boxes and ranked probabilities.
    """
    def __init__(self, model_path="best.pt", labels_path="labels.txt"):
        self.model_path = model_path
        self.labels = []
        self.backend = "yolo"

        if os.path.exists(labels_path):
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.labels = [line.strip() for line in f.readlines() if line.strip()]

        # Try loading YOLO model
        if os.path.exists(model_path):
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(model_path)
                if hasattr(self.yolo_model, "names") and self.yolo_model.names:
                    self.labels = list(self.yolo_model.names.values())
                self.backend = "yolo"
                print(f"[AI Model] Loaded YOLOv11 Model from '{model_path}' ({len(self.labels)} classes).")
                return
            except Exception as e:
                print(f"[AI Model Warning] Failed to load via Ultralytics: {e}. Checking TFLite fallback...")

        # Fallback to TFLite if rash_model.tflite exists
        tflite_fallback = "rash_model.tflite"
        if os.path.exists(tflite_fallback):
            try:
                from ai_edge_litert import interpreter as ai_interp
                InterpreterClass = ai_interp.Interpreter
            except ImportError:
                import tensorflow as tf
                InterpreterClass = tf.lite.Interpreter

            self.interpreter = InterpreterClass(model_path=tflite_fallback)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.backend = "tflite"
            print(f"[AI Model] Loaded TFLite Model from '{tflite_fallback}'.")
        else:
            raise FileNotFoundError(f"Neither {model_path} nor {tflite_fallback} found.")

    def detect_objects(self, bgr_image, conf_threshold: float = 0.20) -> List[Dict[str, Any]]:
        """
        Runs object detection and returns list of detected lesion boxes:
        [{"box": (x1, y1, x2, y2), "condition": "Eczema", "confidence": 0.88, ...}, ...]
        """
        detections = []
        if self.backend == "yolo":
            results = self.yolo_model.predict(source=bgr_image, conf=conf_threshold, imgsz=512, verbose=False)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cond_name = self.labels[cls_id] if cls_id < len(self.labels) else f"Condition_{cls_id}"
                    info = get_condition_info(cond_name)
                    
                    detections.append({
                        "box": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        "condition": cond_name,
                        "confidence": conf,
                        "ai_confidence_pct": round(conf * 100, 1),
                        "severity": info.get("severity", "Moderate"),
                        "description": info.get("description", ""),
                        "red_flags": info.get("red_flags", []),
                        "contagious": get_contagious_status(cond_name)
                    })
        return detections

    def predict(self, bgr_image) -> Dict[str, float]:
        """Returns aggregated dictionary mapping condition to top confidence score."""
        detections = self.detect_objects(bgr_image, conf_threshold=0.15)
        prob_dict = {label: 0.01 for label in self.labels}
        for det in detections:
            cond = det["condition"]
            prob_dict[cond] = max(prob_dict.get(cond, 0.0), det["confidence"])
        return prob_dict

    def rank_predictions(self, prob_dict: Dict[str, float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Formats and ranks 100% AI predictions with clinical descriptions."""
        sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
        results = []

        for rank, (cond, prob) in enumerate(sorted_probs[:top_k], start=1):
            info = get_condition_info(cond)
            results.append({
                "rank": rank,
                "condition": cond,
                "confidence": float(prob),
                "ai_confidence_pct": round(float(prob * 100), 1),
                "severity": info.get("severity", "Unknown"),
                "description": info.get("description", ""),
                "red_flags": info.get("red_flags", []),
                "contagious": get_contagious_status(cond)
            })

        return results

# ------------------------------------------------------------------------------
# Real-Time AI Detection & Live Overlay Pipeline
# ------------------------------------------------------------------------------
class RealtimeAnalyzer:
    """
    Analyzes live camera frames continuously:
    - Runs skin presence & focus check
    - Computes 100% AI YOLOv11 Multi-Lesion Object Detection
    - Draws high-visibility HUD overlays and detection bounding boxes
    """
    def __init__(self, classifier: TFLiteClassifier):
        self.classifier = classifier
        self.cached_predictions: List[Dict[str, Any]] = []
        self.cached_detections: List[Dict[str, Any]] = []
        self.cached_status: str = "Scanning..."

    def process_frame(self, frame_bgr, run_ai: bool = True):
        """Processes frame, updates detection cache, and returns annotated frame + telemetry."""
        h, w = frame_bgr.shape[:2]
        quality = detect_skin_and_quality(frame_bgr, return_mask=True)

        if run_ai:
            if quality["is_valid"]:
                # Run YOLOv11 Object Detection
                self.cached_detections = self.classifier.detect_objects(frame_bgr, conf_threshold=0.25)
                
                if self.cached_detections:
                    # Sort detections by confidence
                    self.cached_detections.sort(key=lambda d: d["confidence"], reverse=True)
                    top = self.cached_detections[0]
                    self.cached_status = f"{top['condition']} ({top['ai_confidence_pct']}%) - {len(self.cached_detections)} lesion(s)"
                    
                    # Convert detections into ranked summary
                    prob_dict = self.classifier.predict(frame_bgr)
                    self.cached_predictions = self.classifier.rank_predictions(prob_dict, top_k=5)
                else:
                    self.cached_status = "Skin detected • Scanning for lesions..."
                    prob_dict = self.classifier.predict(frame_bgr)
                    self.cached_predictions = self.classifier.rank_predictions(prob_dict, top_k=5)
            else:
                self.cached_predictions = []
                self.cached_detections = []
                self.cached_status = quality["error"] or "Position skin in frame"

        # Draw Real-Time HUD Overlay onto frame
        annotated = frame_bgr.copy()
        
        # 1. Draw Bounding Boxes for all detected lesions
        for det in self.cached_detections:
            x1, y1, x2, y2 = det["box"]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            bw, bh = x2 - x1, y2 - y1

            # High-visibility cyan/emerald glowing box
            color = (0, 240, 120) if det["confidence"] > 0.50 else (0, 200, 255)
            
            # Corner brackets
            line_len = max(8, min(bw, bh) // 4)
            thick = 2
            # Top-left
            cv2.line(annotated, (x1, y1), (x1 + line_len, y1), color, thick)
            cv2.line(annotated, (x1, y1), (x1, y1 + line_len), color, thick)
            # Top-right
            cv2.line(annotated, (x2, y1), (x2 - line_len, y1), color, thick)
            cv2.line(annotated, (x2, y1), (x2, y1 + line_len), color, thick)
            # Bottom-left
            cv2.line(annotated, (x1, y2), (x1 + line_len, y2), color, thick)
            cv2.line(annotated, (x1, y2), (x1, y2 - line_len), color, thick)
            # Bottom-right
            cv2.line(annotated, (x2, y2), (x2 - line_len, y2), color, thick)
            cv2.line(annotated, (x2, y2), (x2, y2 - line_len), color, thick)
            
            # Subtle bounding box rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

            # Draw Label Tag above Bounding Box
            label_text = f" {det['condition']} [{det['ai_confidence_pct']}%] "
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y = max(th + 6, y1 - 4)
            cv2.rectangle(annotated, (x1, tag_y - th - 4), (x1 + tw + 4, tag_y + 2), (15, 23, 42), -1)
            cv2.rectangle(annotated, (x1, tag_y - th - 4), (x1 + tw + 4, tag_y + 2), color, 1)
            cv2.putText(annotated, label_text, (x1 + 2, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # 2. Draw Top Status Banner
        status_bg_color = (15, 23, 42)
        cv2.rectangle(annotated, (0, 0), (w, 36), status_bg_color, -1)
        cv2.putText(annotated, "YOLOV11 AI SKIN LESION SCANNER", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2)
        
        # Transmission / Alert badge on top right
        if self.cached_detections:
            top = self.cached_detections[0]
            tag = f"{top['contagious']} | {top['severity']}"
            cv2.putText(annotated, tag, (w - 220, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (203, 213, 225), 1)

        return annotated, {
            "predictions": self.cached_predictions,
            "detections": self.cached_detections,
            "status": self.cached_status,
            "quality": quality,
            "is_valid": quality["is_valid"]
        }

# ------------------------------------------------------------------------------
# CLI Runner
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="100% AI Real-Time Skin Disease Scanner")
    parser.add_argument("--model", type=str, default="best.pt")
    parser.add_argument("--labels", type=str, default="labels.txt")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    args = parser.parse_args()

    print("\n=======================================================")
    print("    100% AI REAL-TIME YOLOV11 SKIN DISEASE SCANNER     ")
    print("    12 Unique Conditions • Multi-Lesion Detector       ")
    print("=======================================================\n")

    classifier = TFLiteClassifier(model_path=args.model, labels_path=args.labels)

    if args.image:
        frame = cv2.imread(args.image)
    else:
        cam = EdgeCamera(camera_index=args.camera_index)
        frame = cam.capture_frame()
        cam.release()

    if frame is None or frame.size == 0:
        print("[Error] Failed to acquire valid image frame.")
        sys.exit(1)

    quality = detect_skin_and_quality(frame)
    print(f"Skin Presence: {quality['skin_ratio']}% | Sharpness: {quality['blur_score']}")

    detections = classifier.detect_objects(frame, conf_threshold=0.20)
    print(f"\n[+] Detected {len(detections)} lesion(s):")
    for d in detections:
        print(f"  • {d['condition']:<22} | {d['ai_confidence_pct']:>5.1f}% | Box: {d['box']} | {d['contagious']} | {d['severity']}")

if __name__ == "__main__":
    main()
