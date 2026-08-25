import os

CLASSES = [
    "Acne_Vulgaris",
    "Chickenpox_Varicella",
    "Eczema",
    "Hives",
    "Impetigo",
    "Melanoma",
    "Psoriasis",
    "Ringworm",
    "Scabies",
    "Warts"
]

dataset_dir = "dataset_boosted"
print("=== DATASET 10 CLASS INVENTORY ===")
total_imgs = 0
for cls in CLASSES:
    cls_path = os.path.join(dataset_dir, cls)
    if os.path.exists(cls_path):
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        print(f"{cls:25}: {len(imgs)} images")
        total_imgs += len(imgs)
    else:
        print(f"{cls:25}: MISSING!")

print(f"\nTotal images for 10 classes: {total_imgs}")
