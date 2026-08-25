import pi_scanner
import cv2
import glob
import numpy as np

clf = pi_scanner.TFLiteClassifier()
print("Model Input Details:", clf.input_details)
print("Model Output Details:", clf.output_details)

print("\n--- TESTING 1 SAMPLE FROM EACH OF THE 10 CLASSES ---")
for cls in clf.labels:
    imgs = glob.glob(f"dataset_10_classes/{cls}/*.*")
    if imgs:
        img = cv2.imread(imgs[0])
        
        # Test A: pi_scanner.predict
        probs_a = clf.predict(img)
        top_a = clf.rank_predictions(probs_a, top_k=1)[0]
        
        # Test B: Raw float32 in [0, 255]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        inp = np.expand_dims(resized.astype(np.float32), axis=0)
        clf.interpreter.set_tensor(clf.input_details[0]['index'], inp)
        clf.interpreter.invoke()
        out = clf.interpreter.get_tensor(clf.output_details[0]['index'])[0]
        top_b_idx = int(np.argmax(out))
        top_b_name = clf.labels[top_b_idx]
        top_b_conf = float(out[top_b_idx])
        
        print(f"Actual: {cls:22} | Preprocessed: {top_a['condition']:20} ({top_a['ai_confidence_pct']:5.1f}%) | Raw: {top_b_name:20} ({top_b_conf*100:5.1f}%)")
