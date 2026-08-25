"""
pi_scanner.py - Edge AI Skin Disease Scanner Engine
Features:
- Real-Time AI Camera Analyzer with live overlay (bounding box / diagnostic badge)
- 100% AI Computer Vision classification across 10 unique dermatological conditions
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

    MIN_SKIN_RATIO = 0.08   # At least 8% skin presence in frame
    MIN_BLUR_SCORE = 30.0   # Sharpness threshold
    MIN_BRIGHTNESS = 20.0
    MAX_BRIGHTNESS = 245.0

    skin_detected = bool(skin_ratio >= MIN_SKIN_RATIO)
    is_sharp = bool(blur_score >= MIN_BLUR_SCORE)
    is_blurry = not is_sharp
    is_poor_lighting = bool(brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS)
    is_valid = bool(skin_detected and is_sharp and not is_poor_lighting)

    error_msg = None
    if not skin_detected:
        error_msg = f"No human skin detected (Coverage: {skin_ratio*100:.1f}%). Please position the camera directly on the skin lesion."
    elif not is_sharp:
        error_msg = f"Camera frame is blurry (Sharpness: {blur_score:.1f} / 30.0 min). Please hold the camera steady."
    elif is_poor_lighting:
        if brightness < MIN_BRIGHTNESS:
            error_msg = "Frame is too dark. Please increase illumination."
        else:
            error_msg = "Frame is over-exposed or glaring."

    warnings = []
    if blur_score < 45.0 and is_sharp:
        warnings.append(f"Moderate sharpness ({blur_score:.1f}). Closer focus provides higher confidence.")

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
# Dynamic TFLite Interpreter Import
# ------------------------------------------------------------------------------
def get_tflite_interpreter_class():
    try:
        mod = importlib.import_module("tflite_runtime.interpreter")
        return getattr(mod, "Interpreter"), "tflite_runtime"
    except ImportError:
        try:
            from ai_edge_litert import interpreter as ai_interp
            return ai_interp.Interpreter, "ai_edge_litert"
        except ImportError:
            try:
                import tensorflow as tf
                return tf.lite.Interpreter, "tensorflow.lite"
            except (ImportError, AttributeError):
                print("[Error] No TFLite backend found.")
                sys.exit(1)

InterpreterClass, TFLITE_BACKEND = get_tflite_interpreter_class()

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
        except Exception as e:
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
# 100% AI TFLite Computer Vision Classifier
# ------------------------------------------------------------------------------
class TFLiteClassifier:
    def __init__(self, model_path="rash_model.tflite", labels_path="labels.txt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        print(f"[TFLite] Loading '{model_path}' ({TFLITE_BACKEND})...")
        self.interpreter = InterpreterClass(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]['shape']
        self.height = self.input_shape[1]
        self.width = self.input_shape[2]

        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = [line.strip() for line in f.readlines() if line.strip()]

        print(f"[TFLite] Ready: {len(self.labels)} classes ({self.width}x{self.height}).")

    def preprocess(self, bgr_image):
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, (self.width, self.height))
        input_data = np.expand_dims(resized_image, axis=0)

        # Model graph contains built-in preprocess_input, so it expects [0, 255] float32
        return input_data.astype(np.float32)

    def predict(self, bgr_image) -> Dict[str, float]:
        """Returns dictionary mapping condition name to 100% AI visual confidence (0.0 to 1.0)."""
        input_data = self.preprocess(bgr_image)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # De-quantize if quantized tensor
        output_type = self.output_details[0]['dtype']
        if output_type in [np.uint8, np.int8]:
            scale, zero_point = self.output_details[0]['quantization']
            output_data = scale * (output_data.astype(np.float32) - zero_point)

        # Apply Softmax if outputs are logits
        if np.sum(output_data) > 1.5 or np.min(output_data) < 0:
            exp_data = np.exp(output_data - np.max(output_data))
            output_data = exp_data / np.sum(exp_data)

        prob_dict = {label: float(output_data[i]) for i, label in enumerate(self.labels)}
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
    - Computes 100% AI Vision prediction
    - Locates rash lesion region of interest (ROI) / bounding box
    - Draws high-visibility HUD overlays and diagnostic target reticles
    """
    def __init__(self, classifier: TFLiteClassifier):
        self.classifier = classifier
        self.last_analysis_time = 0
        self.cached_predictions: List[Dict[str, Any]] = []
        self.cached_bbox: Optional[tuple] = None
        self.cached_status: str = "Scanning..."

    def process_frame(self, frame_bgr, run_ai: bool = True):
        """Processes frame, updates detection cache, and returns annotated frame + telemetry."""
        h, w = frame_bgr.shape[:2]
        quality = detect_skin_and_quality(frame_bgr, return_mask=True)

        if run_ai:
            if quality["is_valid"]:
                probs = self.classifier.predict(frame_bgr)
                self.cached_predictions = self.classifier.rank_predictions(probs, top_k=5)
                
                # Extract Bounding Box / Region of Interest from Skin Mask Contours
                skin_mask = quality.get("skin_mask")
                if skin_mask is not None:
                    # Find largest contour as skin ROI
                    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_c = max(contours, key=cv2.contourArea)
                        if cv2.contourArea(largest_c) > (w * h * 0.05):
                            x, y, bw, bh = cv2.boundingRect(largest_c)
                            self.cached_bbox = (x, y, bw, bh)
                        else:
                            self.cached_bbox = (int(w*0.15), int(h*0.15), int(w*0.7), int(h*0.7))
                else:
                    self.cached_bbox = (int(w*0.15), int(h*0.15), int(w*0.7), int(h*0.7))

                if self.cached_predictions:
                    top = self.cached_predictions[0]
                    self.cached_status = f"{top['condition'].replace('_', ' ')} ({top['ai_confidence_pct']}%)"
            else:
                self.cached_predictions = []
                self.cached_bbox = None
                self.cached_status = quality["error"] or "Position skin in frame"

        # Draw Real-Time HUD Overlay onto frame
        annotated = frame_bgr.copy()
        
        # 1. Draw Target Reticle / Bounding Box
        if self.cached_bbox:
            x, y, bw, bh = self.cached_bbox
            color = (0, 230, 115) if self.cached_predictions and self.cached_predictions[0]["confidence"] > 0.3 else (0, 200, 255)
            
            # Corner brackets
            line_len = min(bw, bh) // 4
            thick = 3
            # Top-left
            cv2.line(annotated, (x, y), (x + line_len, y), color, thick)
            cv2.line(annotated, (x, y), (x, y + line_len), color, thick)
            # Top-right
            cv2.line(annotated, (x + bw, y), (x + bw - line_len, y), color, thick)
            cv2.line(annotated, (x + bw, y), (x + bw, y + line_len), color, thick)
            # Bottom-left
            cv2.line(annotated, (x, y + bh), (x + line_len, y + bh), color, thick)
            cv2.line(annotated, (x, y + bh), (x, y + bh - line_len), color, thick)
            # Bottom-right
            cv2.line(annotated, (x + bw, y + bh), (x + bw - line_len, y + bh), color, thick)
            cv2.line(annotated, (x + bw, y + bh), (x + bw, y + bh - line_len), color, thick)

            # Draw Label Tag above Bounding Box
            if self.cached_predictions:
                top = self.cached_predictions[0]
                label_text = f" AI: {top['condition'].replace('_', ' ')} [{top['ai_confidence_pct']}%] "
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tag_y = max(th + 10, y - 8)
                cv2.rectangle(annotated, (x, tag_y - th - 6), (x + tw + 4, tag_y + 4), (15, 23, 42), -1)
                cv2.rectangle(annotated, (x, tag_y - th - 6), (x + tw + 4, tag_y + 4), color, 1)
                cv2.putText(annotated, label_text, (x + 2, tag_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # 2. Draw Top Status Banner
        status_bg_color = (15, 23, 42)
        cv2.rectangle(annotated, (0, 0), (w, 36), status_bg_color, -1)
        cv2.putText(annotated, "REAL-TIME 100% AI VISION SCANNER", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (56, 189, 248), 2)
        
        # Transmission / Alert badge on top right
        if self.cached_predictions:
            top = self.cached_predictions[0]
            tag = f"{top['contagious']} | {top['severity']}"
            cv2.putText(annotated, tag, (w - 220, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (203, 213, 225), 1)

        return annotated, {
            "predictions": self.cached_predictions,
            "status": self.cached_status,
            "quality": quality,
            "is_valid": quality["is_valid"]
        }

# ------------------------------------------------------------------------------
# CLI Runner
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="100% AI Real-Time Skin Disease Scanner")
    parser.add_argument("--model", type=str, default="rash_model.tflite")
    parser.add_argument("--labels", type=str, default="labels.txt")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()

    print("\n=======================================================")
    print("      100% AI REAL-TIME SKIN DISEASE SCANNER           ")
    print("      10 Unique Conditions • Edge MobileNetV2          ")
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

    probs = classifier.predict(frame)
    top_matches = classifier.rank_predictions(probs, top_k=10)

    print("\n=======================================================")
    print("             TOP 100% AI DIAGNOSTIC MATCHES            ")
    print("=======================================================")
    for rank, match in enumerate(top_matches, start=1):
        print(f"#{rank:<2} | {match['condition']:<25} | {match['ai_confidence_pct']:>5.1f}% AI | {match['contagious']} | {match['severity']}")

if __name__ == "__main__":
    main()
