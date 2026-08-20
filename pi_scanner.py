"""
pi_scanner.py - Raspberry Pi Edge Multimodal Rash Scanner
Integrates Raspberry Pi camera (Picamera2 / OpenCV), TFLite computer vision model,
and clinical symptom matching into a 50/50 Multimodal Fusion Engine.
"""

import os
import sys
import argparse
import time
import numpy as np
import cv2

# Import symptom database & matcher
from symptoms_db import calculate_symptom_score, check_red_flags, get_condition_info, get_contagious_status, SYMPTOM_DB

import importlib

# ------------------------------------------------------------------------------
# Robust TFLite Interpreter Import (tflite_runtime on Pi, tensorflow on PC)
# ------------------------------------------------------------------------------
def get_tflite_interpreter_class():
    """Dynamically loads TFLite Interpreter class for Raspberry Pi or PC."""
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
                print("[Error] No TFLite backend found. Install one of: tflite-runtime, ai-edge-litert, or tensorflow.")
                sys.exit(1)

InterpreterClass, TFLITE_BACKEND = get_tflite_interpreter_class()


# ------------------------------------------------------------------------------
# Robust Camera Module (Picamera2 for Pi 4, OpenCV cv2.VideoCapture fallback)
# ------------------------------------------------------------------------------
class EdgeCamera:
    def __init__(self, camera_index=0, resolution=(640, 480)):
        self.resolution = resolution
        self.picam2 = None
        self.cap = None
        self.backend = None

        # Attempt initializing Picamera2 first (Native Pi 4 Camera)
        try:
            picam_mod = importlib.import_module("picamera2")
            Picamera2_cls = getattr(picam_mod, "Picamera2")
            self.picam2 = Picamera2_cls()
            config = self.picam2.create_preview_configuration(main={"size": resolution})
            self.picam2.configure(config)
            self.picam2.start()
            self.backend = "picamera2"
            print("[Camera] Successfully initialized native Raspberry Pi Picamera2!")
        except Exception as e:

            print(f"[Camera Info] Picamera2 not available ({e}). Falling back to OpenCV VideoCapture...")
            self.cap = cv2.VideoCapture(camera_index)
            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                self.backend = "opencv"
                print(f"[Camera] Initialized OpenCV VideoCapture(index={camera_index}).")
            else:
                self.backend = "synthetic"
                print("[Camera Info] No live physical camera detected. Switched to Synthetic Test Frame mode.")

    def capture_frame(self):
        """Captures a single BGR frame from active camera source."""
        if self.backend == "picamera2" and self.picam2:
            frame_rgb = self.picam2.capture_array()
            # Convert RGB to BGR for uniform OpenCV downstream handling
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        elif self.backend == "opencv" and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        
        # Synthetic Fallback Frame (Color test pattern with skin-tone patch)
        img = np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        cv2.circle(img, (self.resolution[0]//2, self.resolution[1]//2), 100, (140, 180, 210), -1)
        cv2.putText(img, "Synthetic Test Rash Frame", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img

    def release(self):
        """Release camera hardware resources."""
        if self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass
        if self.cap:
            self.cap.release()

# ------------------------------------------------------------------------------
# TFLite Visual Inference Engine
# ------------------------------------------------------------------------------
class TFLiteClassifier:
    def __init__(self, model_path="rash_model.tflite", labels_path="labels.txt"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"TFLite model file not found: {model_path}")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        print(f"[TFLite] Loading model '{model_path}' using backend '{TFLITE_BACKEND}'...")
        self.interpreter = InterpreterClass(model_path=model_path)
        self.interpreter.allocate_tensors()


        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Input shape expected by MobileNetV2 (e.g. 1, 224, 224, 3)
        self.input_shape = self.input_details[0]['shape']
        self.height = self.input_shape[1]
        self.width = self.input_shape[2]

        with open(labels_path, 'r', encoding='utf-8') as f:
            self.labels = [line.strip() for line in f.readlines() if line.strip()]

        print(f"[TFLite] Loaded {len(self.labels)} class labels. Expected input: {self.width}x{self.height}")

    def preprocess(self, bgr_image):
        """Preprocesses camera frame for MobileNetV2 inference."""
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(rgb_image, (self.width, self.height))
        input_data = np.expand_dims(resized_image, axis=0)

        # Scale input to MobileNetV2 [-1, 1] range if float32, or keep uint8 if quantized
        input_type = self.input_details[0]['dtype']
        if input_type == np.float32:
            input_data = (input_data.astype(np.float32) / 127.5) - 1.0

        return input_data

    def predict(self, bgr_image):
        """Runs TFLite model inference on input image and returns probability dictionary."""
        input_data = self.preprocess(bgr_image)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # De-quantize if output tensor is uint8/int8
        output_type = self.output_details[0]['dtype']
        if output_type in [np.uint8, np.int8]:
            scale, zero_point = self.output_details[0]['quantization']
            output_data = scale * (output_data.astype(np.float32) - zero_point)

        # Apply Softmax if outputs are raw logits
        if np.sum(output_data) > 1.5 or np.min(output_data) < 0:
            exp_data = np.exp(output_data - np.max(output_data))
            output_data = exp_data / np.sum(exp_data)

        prob_dict = {label: float(output_data[i]) for i, label in enumerate(self.labels)}
        return prob_dict

# ------------------------------------------------------------------------------
# 50/50 Multimodal Fusion Engine
# ------------------------------------------------------------------------------
def run_multimodal_fusion(visual_probs: dict, user_symptoms: str, top_k=10) -> list:
    """
    Computes a 50/50 Multimodal Fusion score:
      Final Score = 0.50 * (Visual Model Prob) + 0.50 * (Symptom Match Score)
    
    Returns sorted list of Top K matching condition dictionaries.
    """
    fusion_results = []
    
    # Collect all known conditions (from model labels + symptoms DB)
    all_conditions = set(visual_probs.keys()).union(set(SYMPTOM_DB.keys()))

    for condition in all_conditions:
        v_score = visual_probs.get(condition, 0.0)
        s_score = calculate_symptom_score(user_symptoms, condition)
        
        final_score = (0.50 * v_score) + (0.50 * s_score)
        
        info = get_condition_info(condition)
        fusion_results.append({
            "condition": condition,
            "final_score": final_score,
            "visual_score": v_score,
            "symptom_score": s_score,
            "severity": info.get("severity", "Unknown"),
            "description": info.get("description", ""),
            "red_flags": info.get("red_flags", []),
            "contagious": get_contagious_status(condition)
        })

    # Sort descending by Final Multimodal Fusion Score
    fusion_results.sort(key=lambda x: x["final_score"], reverse=True)
    return fusion_results[:top_k]

# ------------------------------------------------------------------------------
# Application Runner & Output Formatter
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Local Multimodal Rash Scanner for Raspberry Pi 4")
    parser.add_argument("--model", type=str, default="rash_model.tflite", help="Path to TFLite model file")
    parser.add_argument("--labels", type=str, default="labels.txt", help="Path to labels text file")
    parser.add_argument("--image", type=str, default=None, help="Path to static image file (bypasses camera)")
    parser.add_argument("--symptoms", type=str, default=None, help="Symptom text query (bypasses interactive prompt)")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera device index")
    parser.add_argument("--test-mode", action="store_true", help="Run quick automated test execution")
    args = parser.parse_args()

    print("\n=======================================================")
    print("      LOCAL MULTIMODAL RASH SCANNER (EDGE AI)          ")
    print("      Raspberry Pi 4 • MobileNetV2 • 50/50 Fusion      ")
    print("=======================================================\n")

    # Load Classifier
    try:
        classifier = TFLiteClassifier(model_path=args.model, labels_path=args.labels)
    except Exception as e:
        print(f"[Fatal Error] Could not initialize TFLite classifier: {e}")
        sys.exit(1)

    # Acquire Image Frame
    if args.image:
        if not os.path.exists(args.image):
            print(f"[Error] Specified image file not found: {args.image}")
            sys.exit(1)
        print(f"[Input] Reading input image file: '{args.image}'...")
        frame = cv2.imread(args.image)
    else:
        camera = EdgeCamera(camera_index=args.camera_index)
        print("[Input] Capturing frame from edge camera module...")
        time.sleep(0.5)  # Warm up sensor
        frame = camera.capture_frame()
        camera.release()

    if frame is None or frame.size == 0:
        print("[Error] Failed to acquire valid image frame.")
        sys.exit(1)

    # Perform Computer Vision Inference
    print("[Vision Engine] Running MobileNetV2 TFLite inference...")
    t0 = time.time()
    visual_probs = classifier.predict(frame)
    inference_time_ms = (time.time() - t0) * 1000
    print(f"[Vision Engine] Inference completed in {inference_time_ms:.1f} ms.")

    # Get User Symptoms Input
    if args.symptoms:
        user_symptoms = args.symptoms
    elif args.test_mode:
        user_symptoms = "itchy dry red skin scaly cracked"
        print(f"[Test Mode] Using sample symptoms: '{user_symptoms}'")
    else:
        print("\n-------------------------------------------------------")
        print("CLINICAL SYMPTOM INPUT")
        print("Describe patient symptoms (e.g. 'itchy dry red skin cracked scaling burning'):")
        user_symptoms = input("> ").strip()

    # Red Flag Warning Check
    red_flag_matches = check_red_flags(user_symptoms)
    if red_flag_matches:
        print("\n" + "!"*60)
        print(" SAFETY WARNING: HIGH-RISK CRITICAL SYMPTOMS DETECTED!")
        print(f" Detected Red Flags: {', '.join(red_flag_matches).upper()}")
        print(" Immediate medical evaluation at an urgent care or emergency room is strongly advised.")
        print("!"*60 + "\n")

    # Run 50/50 Multimodal Fusion Engine
    print("\n[Fusion Engine] Calculating 50/50 Multimodal Fusion Scores...")
    top_matches = run_multimodal_fusion(visual_probs, user_symptoms, top_k=10)

    # Format Results
    print("\n=======================================================")
    print("          TOP 10 RANKED DIAGNOSTIC MATCHES            ")
    print("=======================================================")
    print(f"{'Rank':<5} | {'Condition Name':<30} | {'Match %':<8} | {'Vision %':<8} | {'Symptom %':<10} | {'Severity'}")
    print("-" * 85)

    for rank, match in enumerate(top_matches, start=1):
        cond_name = match["condition"].replace("_", " ")
        final_pct = match["final_score"] * 100
        v_pct = match["visual_score"] * 100
        s_pct = match["symptom_score"] * 100
        severity = match["severity"]
        
        print(f"#{rank:<4} | {cond_name:<30} | {final_pct:>6.1f}% | {v_pct:>6.1f}% | {s_pct:>8.1f}% | {severity}")

    print("\n-------------------------------------------------------")
    print("TOP MATCH CLINICAL SUMMARY:")
    top_cond = top_matches[0]
    print(f" Condition:   {top_cond['condition'].replace('_', ' ')}")
    print(f" Match Score: {top_cond['final_score']*100:.1f}% (50% Vision: {top_cond['visual_score']*100:.1f}%, 50% Symptoms: {top_cond['symptom_score']*100:.1f}%)")
    print(f" Severity:    {top_cond['severity']}")
    print(f" Summary:     {top_cond['description']}")
    if top_cond['red_flags']:
        print(f" Warning Flags: {', '.join(top_cond['red_flags'])}")
    print("-------------------------------------------------------")

    print("\n[DISCLAIMER] This application is an AI research prototype for edge deployment and decision support.")
    print("It is NOT a medical device and should NOT replace professional diagnosis by a qualified dermatologist.\n")

if __name__ == "__main__":
    main()
