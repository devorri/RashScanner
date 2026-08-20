import os
import zipfile
import shutil
import re
import cv2
import numpy as np

# Category normalization mapping to standardized clean folder names
CATEGORY_MAP = {
    "Atopic dermatitist": "Eczema",
    "Uticaria (hives)": "Hives",
    "Tinea Corporis": "Ringworm",
    "Psoriasis": "Psoriasis",
    "Contact Dermatitis": "Contact_Dermatitis",
    "Acne Vulgaris": "Acne_Vulgaris",
    "Herpes Zoster_": "Shingles_Herpes_Zoster",
    "Varicella (chicken pox)": "Chickenpox_Varicella",
    "Hand, Foot, and Mouth Disease (HFMD)_": "Hand_Foot_and_Mouth_Disease",
    "Tinea Pedis": "Athlete_s_Foot_Tinea_Pedis",
    "Tinea Cruris": "Jock_Itch_Tinea_Cruris",
    "Kerosis Polaris (chicken skin)": "Keratosis_Pilaris",
    "Stevens-Johnson Syndrome (SJS)_": "Stevens_Johnson_Syndrome",
    "Toxic Epidermal Necrolysis (TEN)_": "Toxic_Epidermal_Necrolysis",
    "Epidermoid Cyst_": "Epidermoid_Cyst",
    "Sweets Syndrome (acute febrile neutrophilic dermatosis)": "Sweets_Syndrome",
    "Suborrheic Dermatitis": "Seborrheic_Dermatitis",
}

DEFAULT_CATEGORIES = [
    "Eczema",
    "Hives",
    "Ringworm",
    "Psoriasis",
    "Contact_Dermatitis",
    "Acne_Vulgaris",
    "Rosacea",
    "Scabies",
    "Impetigo",
    "Cellulitis",
    "Melanoma",
    "Warts"
]

def sanitize_folder_name(name):
    """Sanitize category name for filesystem and TensorFlow compatibility."""
    name = name.strip()
    if name in CATEGORY_MAP:
        return CATEGORY_MAP[name]
    # Remove non-ascii characters
    clean = name.encode('ascii', 'ignore').decode('ascii')
    clean = re.sub(r'[\t\r\n\f\v]+', '', clean)
    clean = re.sub(r'[^A-Za-z0-9_]+', '_', clean)
    clean = clean.strip('_')
    return clean if clean else "Unknown_Category"

def extract_and_prepare_dataset(zip_paths=None, target_dir="dataset"):
    """
    Extract zip files containing skin condition images into target_dir.
    Ensures pure ASCII filenames to prevent TensorFlow C++ path errors.
    """
    if zip_paths is None:
        zip_paths = [
            "Images of Skin Conditions -20260724T110703Z-1-001.zip",
            "Images of Skin Conditions (continuation)-20260724T110703Z-1-001.zip"
        ]

    # Re-create clean target directory
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)

    # Create default category placeholders
    for default_cat in DEFAULT_CATEGORIES:
        os.makedirs(os.path.join(target_dir, default_cat), exist_ok=True)

    total_extracted = 0
    extracted_categories = set()

    for zip_path in zip_paths:
        if not os.path.exists(zip_path):
            print(f"[Warning] Zip file not found: {zip_path}")
            continue

        print(f"[Info] Unpacking {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            for item in z.infolist():
                if item.is_dir():
                    continue
                parts = [p.strip() for p in item.filename.split('/') if p.strip()]
                if len(parts) >= 2:
                    raw_cat = parts[1]
                    filename = os.path.basename(parts[-1])

                    # Ignore non-image files or hidden files
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        continue

                    clean_cat = sanitize_folder_name(raw_cat)
                    dest_cat_dir = os.path.join(target_dir, clean_cat)
                    os.makedirs(dest_cat_dir, exist_ok=True)
                    
                    extracted_categories.add(clean_cat)

                    with z.open(item) as src_file:
                        file_data = src_file.read()
                    
                    # Verify valid image decoding
                    nparr = np.frombuffer(file_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None or img.size == 0:
                        continue

                    # Save validated image with numbered ASCII filename
                    img_idx = len(os.listdir(dest_cat_dir)) + 1
                    clean_filename = f"image_{img_idx:04d}{ext}"
                    dest_file_path = os.path.join(dest_cat_dir, clean_filename)

                    with open(dest_file_path, 'wb') as dst_file:
                        dst_file.write(file_data)
                    total_extracted += 1


    print(f"\n[Success] Extracted {total_extracted} images across {len(extracted_categories)} categories into '{target_dir}/'.")
    print(f"[Dataset Summary] Total directories in '{target_dir}': {len(os.listdir(target_dir))}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare and extract skin condition dataset.")
    parser.add_argument("--target-dir", type=str, default="dataset", help="Target dataset directory")
    args = parser.parse_args()
    
    extract_and_prepare_dataset(target_dir=args.target_dir)

