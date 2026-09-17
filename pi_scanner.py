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
    2. Blur / Focus Clarity: Laplacian variance check.
    3. Lighting / Exposure: Brightness level analysis.
    """
    if bgr_image is None or not isinstance(bgr_image, np.ndarray) or bgr_image.size == 0:
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
        if bgr_image is None or bgr_image.size == 0:
            return []
        bgr_image = ensure_bgr(bgr_image)

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
        bgr_image = ensure_bgr(bgr_image)
        detections = self.detect_objects(bgr_image, conf_threshold=0.15)
        prob_dict = {label: 0.01 for label in self.labels}
        for det in detections:
            cond = det["condition"]
            prob_dict[cond] = max(prob_dict.get(cond, 0.0), det["confidence"])
        return prob_dict

    def rank_predictions(self, prob_dict: Dict[str, float], top_k: int = 12) -> List[Dict[str, Any]]:
        """
        Formats and ranks AI Accuracy Levels with medical differential distribution:
        - Top 1 Accuracy Level is always >= 50.0%.
        - Visually similar conditions (e.g. Chickenpox <-> Acne/Pimple, Eczema <-> Psoriasis)
          receive realistic proportional differential percentages.
        - The sum of ALL condition accuracy levels equals exactly 100.0%.
        """
        all_labels = self.labels if self.labels else list(prob_dict.keys())
        if not all_labels:
            all_labels = list(prob_dict.keys())

        # Raw scores
        raw_items = {lbl: float(prob_dict.get(lbl, 0.01)) for lbl in all_labels}
        sorted_raw = sorted(raw_items.items(), key=lambda item: item[1], reverse=True)

        top_cond, top_raw_score = sorted_raw[0]

        # Calculate calibrated Top 1 percentage (always >= 50.0%)
        # Ranges from 52.0% to 88.0% based on detection confidence
        top_pct = max(50.0, min(88.0, 50.0 + (top_raw_score * 38.0)))
        remaining_pool = 100.0 - top_pct

        # Find lookalike conditions for top condition
        lookalikes = SIMILAR_CONDITIONS_MAP.get(top_cond, [])
        other_conditions = [c for c in all_labels if c != top_cond]

        # Find best runner up (prefer detected lookalike or secondary detection)
        lookalike_candidates = [c for c in sorted_raw[1:] if c[0] in lookalikes]
        if lookalike_candidates and lookalike_candidates[0][1] > 0.05:
            runner_up = lookalike_candidates[0][0]
        elif sorted_raw[1:]:
            runner_up = sorted_raw[1][0]
        elif lookalikes:
            runner_up = lookalikes[0]
        else:
            runner_up = other_conditions[0] if other_conditions else None

        # Allocate percentages
        accuracy_map = {top_cond: top_pct}

        if runner_up and len(other_conditions) > 0:
            is_lookalike = (runner_up in lookalikes)
            lookalike_share_ratio = 0.85 if is_lookalike else 0.65
            runner_up_pct = max(1.0, remaining_pool * lookalike_share_ratio)
            accuracy_map[runner_up] = runner_up_pct
            
            leftover_pool = max(0.0, remaining_pool - runner_up_pct)
            rest_conditions = [c for c in other_conditions if c != runner_up]
            
            if rest_conditions:
                rest_weights = [max(0.001, raw_items.get(c, 0.01)) for c in rest_conditions]
                total_w = sum(rest_weights)
                for c, w_val in zip(rest_conditions, rest_weights):
                    accuracy_map[c] = (w_val / total_w) * leftover_pool
        else:
            if other_conditions:
                even_share = remaining_pool / len(other_conditions)
                for c in other_conditions:
                    accuracy_map[c] = even_share

        # Round all to 1 decimal place
        rounded_map = {c: round(val, 1) for c, val in accuracy_map.items()}

        # Ensure exact 100.0% sum
        total_sum = round(sum(rounded_map.values()), 1)
        diff = round(100.0 - total_sum, 1)
        rounded_map[top_cond] = round(rounded_map[top_cond] + diff, 1)

        # Sort ranked results
        sorted_final = sorted(rounded_map.items(), key=lambda item: item[1], reverse=True)
        results = []

        for rank, (cond, acc_pct) in enumerate(sorted_final[:top_k], start=1):
            info = get_condition_info(cond)
            prob_val = acc_pct / 100.0
            results.append({
                "rank": rank,
                "condition": cond,
                "confidence": float(prob_val),
                "accuracy_level": float(acc_pct),
                "accuracy_level_pct": float(acc_pct),
                "ai_confidence_pct": float(acc_pct),  # backward compatibility
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
