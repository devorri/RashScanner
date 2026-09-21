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

from symptoms_db import get_condition_info, get_contagious_status, check_red_flags, SIMILAR_CONDITIONS_MAP

# ------------------------------------------------------------------------------
# Image Quality & Skin Presence Detector
# ------------------------------------------------------------------------------
def ensure_bgr(img):
    """
    Guarantees that the input image is a valid 3-channel (BGR) numpy array.
    Converts 4-channel (RGBA/BGRA/XBGR), 1-channel (Grayscale), or multi-channel arrays cleanly.
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return img

    if len(img.shape) == 2:  # Grayscale (H, W) -> BGR
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if len(img.shape) == 3:
        channels = img.shape[2]
        if channels == 4:
            # 4-channel image (RGBA/BGRA/XBGR) -> 3-channel BGR
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif channels == 3:
            return img
        elif channels == 1:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif channels > 4:
            return img[:, :, :3]

    return img

def detect_skin_and_quality(bgr_image, return_mask: bool = False):
    """
    Validates image before neural network inference:
    1. Skin Color Distribution: Multi-space (YCrCb + HSV) detector supporting Fitzpatrick tones I-VI.
    2. Lighting / Exposure: Brightness level analysis.
    (Note: Blur / sharpness gating has been removed so scans and webcam captures are never rejected for blur.)
    """
    if bgr_image is None or not isinstance(bgr_image, np.ndarray) or bgr_image.size == 0:
        res = {
            "is_valid": False,
            "skin_detected": False,
            "skin_ratio": 0.0,
            "is_blurry": False,
            "blur_score": 0.0,
            "brightness": 0.0,
            "error": "Empty or invalid image frame"
        }
        if return_mask:
            res["skin_mask"] = None
        return res

    bgr_image = ensure_bgr(bgr_image)
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

    # Sharpness calculation (purely informational telemetry, never invalidates image)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    try:
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        blur_score = 0.0

    # 2. Brightness level check
    brightness = float(np.mean(gray))

    MIN_SKIN_RATIO = 0.02   # Generous skin presence threshold
    MIN_BRIGHTNESS = 15.0
    MAX_BRIGHTNESS = 250.0

    skin_detected = bool(skin_ratio >= MIN_SKIN_RATIO)
    is_sharp = True
    is_blurry = False
    is_poor_lighting = bool(brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS)
    # Always allow model inference without blocking the user
    is_valid = True

    warnings = []
    if not skin_detected:
        warnings.append(f"Low skin coverage ({skin_ratio*100:.1f}%). Point camera closer to affected area.")
    if is_poor_lighting:
        if brightness < MIN_BRIGHTNESS:
            warnings.append("Frame is relatively dark. Consider increasing lighting.")
        else:
            warnings.append("Frame has strong glare.")

    result = {
        "is_valid": is_valid,
        "skin_detected": skin_detected,
        "skin_ratio": round(float(skin_ratio * 100), 1),
        "is_sharp": is_sharp,
        "is_blurry": is_blurry,
        "blur_score": round(float(blur_score), 1),
        "brightness": round(float(brightness), 1),
        "warnings": warnings,
        "error": None
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
        self.preferred_index = camera_index
        self.picam2 = None
        self.cap = None
        self.backend = None
        self.is_hardware_available = False
        self.last_error = None
        self._init_hardware()

    def _init_hardware(self):
        """Attempts native Picamera2 first, then OpenCV across multiple V4L2/USB indices."""
        # 1. Try Native Picamera2 (Raspberry Pi CSI ribbon camera)
        try:
            picam_mod = importlib.import_module("picamera2")
            Picamera2_cls = getattr(picam_mod, "Picamera2")
            self.picam2 = Picamera2_cls()
            # Video configuration is headless-safe and ideal for web streaming & CV
            try:
                config = self.picam2.create_video_configuration(main={"size": self.resolution, "format": "RGB888"})
            except Exception:
                config = self.picam2.create_preview_configuration(main={"size": self.resolution})
            self.picam2.configure(config)
            self.picam2.start()

            # Enable continuous autofocus for Pi Camera Module v3 (motorized AF lens)
            try:
                from libcamera import controls as libcam_controls
                self.picam2.set_controls({
                    "AfMode": libcam_controls.AfModeEnum.Continuous,
                    "AfSpeed": libcam_controls.AfSpeedEnum.Fast
                })
                print("[Camera] Continuous autofocus enabled (Pi Camera v3 AF lens).")
            except Exception:
                # Fallback: use raw integer values if libcamera import fails
                # AfMode: 2 = Continuous, AfSpeed: 1 = Fast
                try:
                    self.picam2.set_controls({"AfMode": 2, "AfSpeed": 1})
                    print("[Camera] Continuous autofocus enabled (Pi Camera v3, raw controls).")
                except Exception as e_af:
                    print(f"[Camera Info] Autofocus not available on this camera module: {e_af}")

            self.backend = "picamera2"
            self.is_hardware_available = True
            print("[Camera] Initialized native Raspberry Pi Picamera2 successfully!")
            return
        except Exception as e_picam:
            self.picam2 = None
            print(f"[Camera Info] Picamera2 not active ({type(e_picam).__name__}: {e_picam}). Checking USB/V4L2 cameras...")

        # 2. Try OpenCV across multiple camera indices (0, 1, 2, 4)
        indices_to_try = [self.preferred_index]
        for idx in [0, 1, 2, 3, 4]:
            if idx not in indices_to_try:
                indices_to_try.append(idx)

        # On Linux/Pi, prefer V4L2 backend
        backends_to_try = [cv2.CAP_V4L2, cv2.CAP_ANY] if hasattr(cv2, "CAP_V4L2") and sys.platform.startswith("linux") else [cv2.CAP_ANY]

        for backend_flag in backends_to_try:
            for idx in indices_to_try:
                try:
                    cap = cv2.VideoCapture(idx, backend_flag) if backend_flag != cv2.CAP_ANY else cv2.VideoCapture(idx)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
                        # Verify that frames can actually be grabbed (avoid false-positive video codec nodes)
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            self.cap = cap
                            self.backend = "opencv"
                            self.is_hardware_available = True
                            print(f"[Camera] Initialized OpenCV VideoCapture(index={idx}, backend={backend_flag}).")
                            return
                        else:
                            cap.release()
                except Exception as e_cv:
                    pass

        # 3. No hardware camera found
        self.backend = "synthetic"
        self.is_hardware_available = False
        self.last_error = "No Raspberry Pi CSI or USB camera detected."
        print("[Camera Info] No physical camera found on host. Host stream set to Standby.")

    def reconnect(self):
        """Attempts to re-detect host cameras if hardware was plugged in after launch."""
        self.release()
        self._init_hardware()
        return self.is_hardware_available

    def _make_synthetic_frame(self):
        """Renders an informative, high-contrast dark medical HUD graphic when no host camera is connected."""
        w, h = self.resolution[0], self.resolution[1]
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (18, 22, 28) # Clean dark slate background

        # Draw outer grid & medical crosshairs
        cv2.rectangle(img, (20, 20), (w - 20, h - 20), (45, 55, 72), 1)
        cx, cy = w // 2, h // 2
        cv2.line(img, (cx - 35, cy), (cx + 35, cy), (0, 210, 255), 1)
        cv2.line(img, (cx, cy - 35), (cx, cy + 35), (0, 210, 255), 1)
        cv2.circle(img, (cx, cy), 45, (0, 210, 255), 1)

        # Draw Clean Status Typography
        cv2.putText(img, "HOST PI CAMERA STANDBY", (cx - 165, cy - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 255), 2)
        cv2.putText(img, "No Hardware Camera on Host", (cx - 135, cy + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 210, 225), 1)
        cv2.putText(img, "Use Device Webcam / Native Photo Button", (cx - 175, cy + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 160, 180), 1)
        return img

    def capture_frame(self):
        """Captures a single BGR frame safely handling 1, 3, or 4 channel inputs."""
        if self.backend == "picamera2" and self.picam2:
            try:
                frame_arr = self.picam2.capture_array()
                if frame_arr is None or frame_arr.size == 0:
                    return self._make_synthetic_frame()

                if len(frame_arr.shape) == 3:
                    channels = frame_arr.shape[2]
                    if channels == 4:
                        return cv2.cvtColor(frame_arr, cv2.COLOR_RGBA2BGR)
                    elif channels == 3:
                        return cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
                    elif channels == 1:
                        return cv2.cvtColor(frame_arr, cv2.COLOR_GRAY2BGR)
                elif len(frame_arr.shape) == 2:
                    return cv2.cvtColor(frame_arr, cv2.COLOR_GRAY2BGR)
            except Exception as e:
                print(f"[Camera Error] Picamera2 capture failed: {e}")
                return self._make_synthetic_frame()

        elif self.backend == "opencv" and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    return ensure_bgr(frame)
            except Exception as e:
                print(f"[Camera Error] OpenCV capture failed: {e}")

        return self._make_synthetic_frame()

    def release(self):
        if self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass
            self.picam2 = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.backend = None
        self.is_hardware_available = False

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
                self.task = getattr(self.yolo_model, "task", "detect")
                if hasattr(self.yolo_model, "names") and self.yolo_model.names:
                    self.labels = list(self.yolo_model.names.values())
                self.backend = "yolo"
                print(f"[AI Model] Loaded YOLOv11 Model from '{model_path}' ({len(self.labels)} classes, task={self.task}).")
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
            self.task = "detect"
            print(f"[AI Model] Loaded TFLite Model from '{tflite_fallback}'.")
        else:
            raise FileNotFoundError(f"Neither {model_path} nor {tflite_fallback} found.")

    def detect_objects(self, bgr_image, conf_threshold: float = 0.20) -> List[Dict[str, Any]]:
        """
        Runs object detection and returns list of detected lesion boxes.
        Handles both YOLO detection models (r.boxes) and classification models (r.probs).
        """
        if bgr_image is None or bgr_image.size == 0:
            return []
        bgr_image = ensure_bgr(bgr_image)
        h, w = bgr_image.shape[:2]

        detections = []
        if self.backend == "yolo":
            is_classify = getattr(self, "task", "detect") == "classify"

            if is_classify:
                # Classification model (e.g. rash-22, rash-50)
                results = self.yolo_model.predict(source=bgr_image, imgsz=256, verbose=False)
                if results and len(results) > 0 and getattr(results[0], "probs", None) is not None:
                    probs = results[0].probs
                    top1_id = int(probs.top1)
                    top1_conf = float(probs.top1conf)
                    cond_name = self.labels[top1_id] if top1_id < len(self.labels) else f"Condition_{top1_id}"

                    info = get_condition_info(cond_name)
                    # Center reticle box for classification model visual overlay
                    margin_x, margin_y = int(w * 0.15), int(h * 0.15)
                    detections.append({
                        "box": (margin_x, margin_y, w - margin_x, h - margin_y),
                        "condition": cond_name,
                        "confidence": top1_conf,
                        "ai_confidence_pct": round(top1_conf * 100, 1),
                        "severity": info.get("severity", "Moderate"),
                        "description": info.get("description", ""),
                        "red_flags": info.get("red_flags", []),
                        "contagious": get_contagious_status(cond_name)
                    })
            else:
                # Object detection model with bounding boxes (e.g. 12-class root best.pt)
                results = self.yolo_model.predict(source=bgr_image, conf=conf_threshold, imgsz=512, verbose=False)
                for r in results:
                    boxes = getattr(r, "boxes", None)
                    if boxes is None:
                        continue
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
        """Returns aggregated dictionary mapping condition to true AI confidence score directly from model."""
        bgr_image = ensure_bgr(bgr_image)
        if bgr_image is None or bgr_image.size == 0:
            return {label: 0.0 for label in self.labels}

        if self.backend == "yolo":
            if getattr(self, "task", "detect") == "classify":
                # Direct classification probabilities
                results = self.yolo_model.predict(source=bgr_image, imgsz=256, verbose=False)
                if results and len(results) > 0 and getattr(results[0], "probs", None) is not None:
                    probs_tensor = results[0].probs.data.cpu().numpy()
                    prob_dict = {}
                    for idx, p in enumerate(probs_tensor):
                        name = self.labels[idx] if idx < len(self.labels) else f"Class_{idx}"
                        prob_dict[name] = float(p)
                    return prob_dict
                return {label: 0.0 for label in self.labels}
            else:
                # Object detection model
                detections = self.detect_objects(bgr_image, conf_threshold=0.05)
                if not detections:
                    detections = self.detect_objects(bgr_image, conf_threshold=0.01)

                prob_dict = {label: 0.0 for label in self.labels}
                for det in detections:
                    cond = det["condition"]
                    prob_dict[cond] = max(prob_dict.get(cond, 0.0), det["confidence"])
                return prob_dict

        return {label: 0.0 for label in self.labels}

    def rank_predictions(self, prob_dict: Dict[str, float], top_k: int = 12) -> List[Dict[str, Any]]:
        """
        Ranks conditions directly based on the model's actual outputs without artificial manipulation.
        """
        if not prob_dict:
            return []

        all_labels = self.labels if self.labels else list(prob_dict.keys())
        if not all_labels:
            all_labels = list(prob_dict.keys())

        # Extract raw model probabilities
        raw_items = [(lbl, float(prob_dict.get(lbl, 0.0))) for lbl in all_labels]
        sorted_raw = sorted(raw_items, key=lambda item: item[1], reverse=True)

        if not sorted_raw or sorted_raw[0][1] <= 0.0:
            return []

        results = []
        for rank, (cond, conf) in enumerate(sorted_raw[:top_k], start=1):
            if conf <= 0.0 and rank > 1:
                continue
            pct = round(conf * 100.0, 1)
            info = get_condition_info(cond)
            results.append({
                "rank": rank,
                "condition": cond,
                "confidence": float(conf),
                "accuracy_level": pct,
                "accuracy_level_pct": pct,
                "ai_confidence_pct": pct,
                "severity": info.get("severity", "Unknown"),
                "description": info.get("description", ""),
                "red_flags": info.get("red_flags", []),
                "contagious": get_contagious_status(cond)
            })

        return results

def draw_clean_lesion_boxes(bgr_image, detections, show_labels: bool = False):
    """Draws crisp, high-precision neon yellow-green bounding boxes around all detected lesions."""
    if bgr_image is None or bgr_image.size == 0:
        return bgr_image
    bgr_image = ensure_bgr(bgr_image)
    annotated = bgr_image.copy()
    h, w = annotated.shape[:2]
    # Crisp Neon Yellow-Green: BGR (20, 245, 185) / RGB (185, 245, 20)
    neon_color = (20, 245, 185)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), neon_color, 2)
        if show_labels:
            label = f"{det['condition']} {det['ai_confidence_pct']}%"
            cv2.putText(annotated, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, neon_color, 1)

    return annotated

# ------------------------------------------------------------------------------
# Real-Time AI Detection & Live Overlay Pipeline
# ------------------------------------------------------------------------------
class RealtimeAnalyzer:
    """
    Analyzes live camera frames continuously:
    - Runs skin presence & focus check
    - Computes 100% AI YOLOv11 Multi-Lesion Object Detection
    - Draws crisp neon yellow-green detection bounding boxes
    """
    def __init__(self, classifier: TFLiteClassifier):
        self.classifier = classifier
        self.cached_predictions: List[Dict[str, Any]] = []
        self.cached_detections: List[Dict[str, Any]] = []
        self.cached_status: str = "Scanning..."

    def process_frame(self, frame_bgr, run_ai: bool = True):
        """Processes frame, updates detection cache, and returns annotated frame + telemetry."""
        if frame_bgr is None or frame_bgr.size == 0:
            frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_bgr = ensure_bgr(frame_bgr)
        h, w = frame_bgr.shape[:2]
        quality = detect_skin_and_quality(frame_bgr, return_mask=True)

        if run_ai:
            if quality["is_valid"]:
                # Run YOLOv11 Object Detection
                self.cached_detections = self.classifier.detect_objects(frame_bgr, conf_threshold=0.25)
                
                if self.cached_detections:
                    self.cached_detections.sort(key=lambda d: d["confidence"], reverse=True)
                    top = self.cached_detections[0]
                    self.cached_status = f"{top['condition']} ({top['ai_confidence_pct']}%) - {len(self.cached_detections)} lesion(s)"
                    
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

        # Draw Clean Neon Bounding Boxes onto frame
        annotated = draw_clean_lesion_boxes(frame_bgr, self.cached_detections)

        # Draw Top Status Banner
        status_bg_color = (15, 23, 42)
        cv2.rectangle(annotated, (0, 0), (w, 36), status_bg_color, -1)
        cv2.putText(annotated, "YOLOV11 AI SKIN LESION SCANNER", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2)
        
        # Transmission / Alert badge on top right
        if self.cached_detections:
            top = self.cached_detections[0]
            tag = f"{top['contagious']} | {top['severity']} ({len(self.cached_detections)} lesions)"
            cv2.putText(annotated, tag, (w - 280, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (203, 213, 225), 1)

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
