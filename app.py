"""
app.py - Local Multimodal Rashilience Web Application & Admin Server
Integrates Flask, SQLite database, TFLite edge model, Picamera2 / OpenCV live camera,
and complete Patient Clinical Assessment management.
"""

import os
import sys
import json
import sqlite3
import datetime
import base64
import uuid
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, session, send_from_directory

# Import AI Engine components
from pi_scanner import TFLiteClassifier, run_multimodal_fusion, EdgeCamera
from symptoms_db import check_red_flags, get_condition_info

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "rash_scanner_secret_key_edge_pi"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
DB_PATH = os.path.join(os.path.dirname(__file__), "patients.db")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "rash_model.tflite")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels.txt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Lazy-loaded TFLite Classifier Instance
classifier = None

def get_classifier():
    global classifier
    if classifier is None:
        if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
            try:
                classifier = TFLiteClassifier(model_path=MODEL_PATH, labels_path=LABELS_PATH)
                print("[App] TFLite classifier initialized successfully.")
            except Exception as e:
                print(f"[App Warning] Failed to initialize TFLite classifier: {e}")
        else:
            print("[App Warning] rash_model.tflite or labels.txt not found.")
    return classifier

# ------------------------------------------------------------------------------
# SQLite Database Management
# ------------------------------------------------------------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users Table (Admin authentication)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)

    # Seed Admin User (admin / admin123)
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (username, password) VALUES ('admin', 'admin123')")
        print("[DB] Admin user created: admin / admin123")

    # Patients Table (Clinical Intake & Assessment History)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_of_assessment TEXT,
        assessed_by TEXT,
        patient_name TEXT NOT NULL,
        age INTEGER,
        sex TEXT,
        background TEXT,
        current_residence TEXT,
        onset TEXT,
        pattern TEXT,
        progression TEXT,
        location TEXT,
        provoking_relieving_factors TEXT,
        associated_symptoms TEXT,
        treatment_history TEXT,
        past_medical_history TEXT,
        family_history TEXT,
        occupational_hobbies TEXT,
        travel TEXT,
        drug_history TEXT,
        smoking_alcohol TEXT,
        allergies TEXT,
        psychological_social TEXT,
        distribution TEXT,
        color_discoloration TEXT,
        morphology TEXT,
        regional_lymph_nodes TEXT,
        primary_diagnosis TEXT,
        ddx_1 TEXT,
        ddx_2 TEXT,
        ddx_3 TEXT,
        plan_investigations TEXT,
        plan_management TEXT,
        plan_referral TEXT,
        image_filename TEXT,
        ai_results_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def save_base64_image(base64_str):
    """Saves a base64 encoded image string to uploads directory."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    filename = f"capture_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, "wb") as f:
        f.write(img_bytes)
    return filename

# ------------------------------------------------------------------------------
# REST API Routes
# ------------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/uploads/<filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, username FROM users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        session["admin_logged_in"] = True
        session["username"] = user[1]
        return jsonify({"success": True, "message": "Login successful", "username": user[1]})
    else:
        return jsonify({"success": False, "message": "Invalid username or password"}), 401

@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route("/api/camera/snap", methods=["GET"])
def camera_snap():
    """Captures a live snapshot from Raspberry Pi Camera (Picamera2 / OpenCV fallback)."""
    try:
        cam = EdgeCamera()
        frame_bgr = cam.capture_frame()
        cam.release()

        if frame_bgr is None or frame_bgr.size == 0:
            return jsonify({"success": False, "message": "Failed to capture frame from camera"}), 500

        # Encode to JPEG base64
        _, buffer = cv2.imencode('.jpg', frame_bgr)
        base64_img = base64.b64encode(buffer).decode('utf-8')
        
        # Save snapshot file
        filename = f"cam_snap_{uuid.uuid4().hex[:8]}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        cv2.imwrite(filepath, frame_bgr)

        return jsonify({
            "success": True,
            "filename": filename,
            "image_url": f"/uploads/{filename}",
            "base64": f"data:image/jpeg;base64,{base64_img}"
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/examine", methods=["POST"])
def examine_rash():
    """
    Multimodal Examination Endpoint:
    Accepts uploaded file OR base64 image string + symptoms text.
    Runs TFLite model + 50/50 Multimodal Fusion Engine.
    """
    try:
        clf = get_classifier()
        if not clf:
            return jsonify({"success": False, "message": "TFLite AI classifier not initialized. Check model files."}), 500

        symptoms_text = request.form.get("associated_symptoms", "")
        if not symptoms_text:
            symptoms_text = request.form.get("symptoms", "")

        filename = None
        frame_bgr = None

        # Check uploaded file
        if "image_file" in request.files and request.files["image_file"].filename != "":
            file = request.files["image_file"]
            ext = os.path.splitext(file.filename)[1].lower()
            filename = f"upload_{uuid.uuid4().hex[:8]}{ext}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            frame_bgr = cv2.imread(filepath)
        # Check base64 image payload (e.g. from camera)
        elif request.form.get("image_base64"):
            base64_str = request.form.get("image_base64")
            filename = save_base64_image(base64_str)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            frame_bgr = cv2.imread(filepath)
        # Check existing filename reference
        elif request.form.get("image_filename"):
            filename = request.form.get("image_filename")
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.exists(filepath):
                frame_bgr = cv2.imread(filepath)

        if frame_bgr is None or frame_bgr.size == 0:
            return jsonify({"success": False, "message": "Please upload an image or capture a camera frame for analysis."}), 400

        # Run TFLite Computer Vision Model
        t0 = datetime.datetime.now()
        visual_probs = clf.predict(frame_bgr)
        elapsed_ms = (datetime.datetime.now() - t0).total_seconds() * 1000

        # Run 50/50 Multimodal Fusion Engine
        top_matches = run_multimodal_fusion(visual_probs, symptoms_text, top_k=10)
        red_flags = check_red_flags(symptoms_text)

        # Primary & Differential Diagnosis Auto-suggestions
        primary_diag = top_matches[0]["condition"].replace("_", " ") if top_matches else "Inconclusive"
        ddx_1 = top_matches[1]["condition"].replace("_", " ") if len(top_matches) > 1 else ""
        ddx_2 = top_matches[2]["condition"].replace("_", " ") if len(top_matches) > 2 else ""
        ddx_3 = top_matches[3]["condition"].replace("_", " ") if len(top_matches) > 3 else ""

        return jsonify({
            "success": True,
            "image_filename": filename,
            "image_url": f"/uploads/{filename}",
            "inference_time_ms": round(elapsed_ms, 1),
            "top_matches": top_matches,
            "red_flags": red_flags,
            "suggestions": {
                "primary_diagnosis": primary_diag,
                "ddx_1": ddx_1,
                "ddx_2": ddx_2,
                "ddx_3": ddx_3
            }
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/api/patients", methods=["GET", "POST"])
def manage_patients():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if request.method == "POST":
        data = request.get_json() or {}
        
        patient_name = data.get("patient_name", "").strip()
        if not patient_name:
            conn.close()
            return jsonify({"success": False, "message": "Patient Name is required."}), 400

        cursor.execute("""
        INSERT INTO patients (
            date_of_assessment, assessed_by, patient_name, age, sex, background, current_residence,
            onset, pattern, progression, location, provoking_relieving_factors, associated_symptoms, treatment_history,
            past_medical_history, family_history, occupational_hobbies, travel, drug_history, smoking_alcohol, allergies, psychological_social,
            distribution, color_discoloration, morphology, regional_lymph_nodes,
            primary_diagnosis, ddx_1, ddx_2, ddx_3,
            plan_investigations, plan_management, plan_referral,
            image_filename, ai_results_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data.get("date_of_assessment", datetime.date.today().isoformat()),
            data.get("assessed_by", ""),
            patient_name,
            data.get("age"),
            data.get("sex", "female"),
            data.get("background", ""),
            data.get("current_residence", ""),
            data.get("onset", "Sudden/Acute"),
            data.get("pattern", "Constant"),
            data.get("progression", "Static"),
            data.get("location", ""),
            data.get("provoking_relieving_factors", ""),
            data.get("associated_symptoms", ""),
            data.get("treatment_history", ""),
            data.get("past_medical_history", ""),
            data.get("family_history", ""),
            data.get("occupational_hobbies", ""),
            data.get("travel", ""),
            data.get("drug_history", ""),
            data.get("smoking_alcohol", ""),
            data.get("allergies", ""),
            data.get("psychological_social", ""),
            data.get("distribution", ""),
            data.get("color_discoloration", ""),
            data.get("morphology", ""),
            data.get("regional_lymph_nodes", "No"),
            data.get("primary_diagnosis", ""),
            data.get("ddx_1", ""),
            data.get("ddx_2", ""),
            data.get("ddx_3", ""),
            data.get("plan_investigations", ""),
            data.get("plan_management", ""),
            data.get("plan_referral", ""),
            data.get("image_filename", ""),
            json.dumps(data.get("ai_results", []))
        ))

        patient_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return jsonify({"success": True, "message": "Patient record saved successfully", "patient_id": patient_id})

    else:
        # GET: Fetch all patient records for Admin Dashboard
        cursor.execute("SELECT * FROM patients ORDER BY created_at DESC")
        rows = cursor.fetchall()
        patients = [dict(row) for row in rows]
        conn.close()
        return jsonify({"success": True, "patients": patients})

@app.route("/api/patients/<int:patient_id>", methods=["GET", "DELETE"])
def patient_detail(patient_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if request.method == "DELETE":
        cursor.execute("DELETE FROM patients WHERE id = ?", (patient_id,))
        conn.commit()
        conn.close()
        return jsonify({"success": True, "message": f"Patient record #{patient_id} deleted."})

    else:
        cursor.execute("SELECT * FROM patients WHERE id = ?", (patient_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            patient_data = dict(row)
            if patient_data.get("ai_results_json"):
                try:
                    patient_data["ai_results"] = json.loads(patient_data["ai_results_json"])
                except Exception:
                    patient_data["ai_results"] = []
            return jsonify({"success": True, "patient": patient_data})
        else:
            return jsonify({"success": False, "message": "Patient record not found."}), 404

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Local Multimodal Rash Scanner Web App Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run web server on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP to bind web server")
    args = parser.parse_args()

    print(f"\n=======================================================")
    print(f"   RASHILIENCE WEB SERVER RUNNING ON PORT {args.port} ")
    print(f"   Access Web Portal: http://localhost:{args.port}")
    print(f"   Admin Login: admin / admin123")
    print(f"=======================================================\n")
    
    app.run(host=args.host, port=args.port, debug=False)
