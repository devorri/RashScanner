import glob
import cv2
import numpy as np
import pi_scanner

clf = pi_scanner.TFLiteClassifier()

print("="*65)
print(f"{'Condition':<24} | {'Tested':<6} | {'Correct':<7} | {'Accuracy %'}")
print("="*65)

total_tested = 0
total_correct = 0

for cls in clf.labels:
    imgs = glob.glob(f"dataset_10_classes/{cls}/*.*")
    # Take 25 test samples per class
    sample_imgs = imgs[:25]
    correct = 0
    
    for img_p in sample_imgs:
        img = cv2.imread(img_p)
        if img is None:
            continue
        probs = clf.predict(img)
        top = clf.rank_predictions(probs, top_k=1)[0]
        if top["condition"].lower() == cls.lower():
            correct += 1
            
    acc = (correct / len(sample_imgs)) * 100 if sample_imgs else 0
    total_tested += len(sample_imgs)
    total_correct += correct
    print(f"{cls:<24} | {len(sample_imgs):<6} | {correct:<7} | {acc:6.1f}%")

overall_acc = (total_correct / total_tested) * 100
print("="*65)
print(f"OVERALL 10-CLASS ACCURACY: {total_correct}/{total_tested} ({overall_acc:.1f}%)")
