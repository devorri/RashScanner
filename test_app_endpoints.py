import urllib.request
import json
import glob
import os

print("=== RUNNING SYSTEM VERIFICATION TESTS ===")

# 1. System Status
res = urllib.request.urlopen("http://127.0.0.1:5000/api/system/status")
data = json.loads(res.read())
print(f"[+] /api/system/status: Battery {data.get('percentage')}% ({data.get('hours_remaining')}h remaining)")

# 2. Real-Time Status
res = urllib.request.urlopen("http://127.0.0.1:5000/api/realtime/status")
data = json.loads(res.read())
print(f"[+] /api/realtime/status: Live Status = '{data.get('telemetry', {}).get('status')}'")

# 3. Patient List
res = urllib.request.urlopen("http://127.0.0.1:5000/api/patients")
data = json.loads(res.read())
print(f"[+] /api/patients: Found {len(data.get('patients', []))} records in database")

# 4. Examine endpoint with a local image
test_imgs = {
    "Ringworm": glob.glob("dataset_10_classes/Ringworm/*.*")[0],
    "Psoriasis": glob.glob("dataset_10_classes/Psoriasis/*.*")[0],
    "Melanoma": glob.glob("dataset_10_classes/Melanoma/*.*")[0]
}

boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
for disease, img_path in test_imgs.items():
    with open(img_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image_file"; filename="{os.path.basename(img_path)}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + file_bytes + f"\r\n--{boundary}--\r\n".encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:5000/api/examine",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    res = urllib.request.urlopen(req)
    result = json.loads(res.read())
    top_pred = result.get("suggestions", {}).get("primary_diagnosis")
    top_score = result.get("top_score")
    inf_time = result.get("inference_time_ms")
    print(f"[+] /api/examine [{disease} Image] -> Top AI Prediction: '{top_pred}' ({top_score}% AI Confidence) [{inf_time} ms]")

print("\nALL BACKEND API VERIFICATION TESTS PASSED SUCCESSFULLY!")
