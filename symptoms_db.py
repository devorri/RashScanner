"""
symptoms_db.py - Clinical Information & Safety Database for 12 Skin Conditions
Provides clinical descriptions, transmission classification (Contact vs Non-Contact),
and critical emergency Red-Flag safety checks.
Note: AI diagnosis is 100% Computer Vision-driven with YOLOv11 Object Detection.
"""

from typing import Dict, List, Any

# Clinical Reference Information for the 12 Visually-Distinct Conditions
SYMPTOM_DB: Dict[str, Dict[str, Any]] = {
    "Acne": {
        "symptoms": ["pimples", "blackheads", "whiteheads", "oily", "red bumps", "pustules", "nodules", "tender", "cysts"],
        "severity": "Mild to Moderate",
        "description": "Clogged hair follicles with oil and dead skin cells causing inflammatory papules, pustules, or cysts.",
        "red_flags": ["deep painful cysts", "scarring", "fever", "sudden explosive onset"]
    },
    "Pimple": {
        "symptoms": ["single red bump", "pus-filled head", "inflamed pore", "localized tenderness"],
        "severity": "Mild (Benign)",
        "description": "Localized inflammatory pustule or comedone caused by sebum and bacterial buildup in pores.",
        "red_flags": ["facial cellulitis", "rapidly spreading redness"]
    },
    "Chickenpox": {
        "symptoms": ["itchy blisters", "red spots", "dewdrops on rose petal", "fever", "fatigue", "crusting", "widespread"],
        "severity": "Moderate",
        "description": "Highly contagious varicella-zoster viral infection causing itchy fluid-filled vesicles and crusts.",
        "red_flags": ["difficulty breathing", "high fever", "neurological symptoms", "secondary skin infection"]
    },
    "Monkeypox": {
        "symptoms": ["umbilicated pustules", "firm lesions", "swollen lymph nodes", "fever", "body aches", "rash on palms/soles"],
        "severity": "Moderate to High",
        "description": "Orthopoxvirus infection presenting with characteristic deep-seated, well-circumscribed lesions with central umbilication.",
        "red_flags": ["ocular involvement / eye pain", "difficulty breathing", "encephalitis signs", "severe secondary infection"]
    },
    "Eczema": {
        "symptoms": ["itchy", "dry", "red", "inflamed", "cracked", "rough", "scaling", "sensitive", "leathery"],
        "severity": "Mild to Moderate",
        "description": "Atopic Dermatitis causing dry, intensely itchy, inflamed, and scaling skin patches.",
        "red_flags": ["pus", "yellow crust", "fever", "severe pain", "spreading redness"]
    },
    "Psoriasis": {
        "symptoms": ["silvery scales", "thickened", "red patches", "dry", "cracked", "itching", "burning", "plaque", "stiff joints"],
        "severity": "Moderate to Severe",
        "description": "Autoimmune condition marked by rapid skin cell buildup forming thick silvery-white scaly plaques on red base.",
        "red_flags": ["joint swelling", "joint pain", "coverage over 90% body", "fever", "chills"]
    },
    "Ringworm": {
        "symptoms": ["circular", "ring-shaped", "itchy", "red", "scaly", "raised border", "clear center", "spreading"],
        "severity": "Mild",
        "description": "Tinea Corporis fungal infection producing distinct circular ring-shaped scaly red patches with clear centers.",
        "red_flags": ["spreading to face", "scalp involvement", "pus", "secondary bacterial infection"]
    },
    "tinea-versicolor": {
        "symptoms": ["discolored patches", "lighter or darker skin patches", "mild scaling", "trunk", "back", "chest", "sweating"],
        "severity": "Mild (Benign)",
        "description": "Fungal skin infection (Pityrosporum) causing small, discolored, scaly patches on upper trunk and arms.",
        "red_flags": ["secondary infection", "severe burning sensation"]
    },
    "vitiligo": {
        "symptoms": ["depigmented white patches", "loss of skin color", "symmetrical white spots", "premature grey hair"],
        "severity": "Chronic / Cosmetic",
        "description": "Autoimmune disorder where melanocytes are destroyed, causing smooth milky-white patches of depigmentation.",
        "red_flags": ["rapid spread", "co-occurring thyroid/autoimmune crisis"]
    },
    "basal cell carcinoma": {
        "symptoms": ["pearly translucent bump", "visible blood vessels", "rolled border", "non-healing sore", "sun-exposed skin"],
        "severity": "High (Malignant)",
        "description": "Most common form of skin cancer arising in basal cells; presents as pearly papules or bleeding non-healing ulcers.",
        "red_flags": ["rapid growth", "frequent ulceration and bleeding", "proximity to eyes or nose"]
    },
    "melanoma": {
        "symptoms": ["asymmetrical mole", "irregular border", "color variation", "diameter > 6mm", "evolving mole", "dark spot"],
        "severity": "Emergency (Malignancy)",
        "description": "High-risk aggressive skin malignancy arising from melanocytes; exhibits ABCDE criteria.",
        "red_flags": ["bleeding mole", "rapid evolution", "new black lesion", "itching or pain in mole"]
    },
    "warts": {
        "symptoms": ["rough bump", "verrucous surface", "black pinpoint dots", "hands", "feet", "plantar", "verruca"],
        "severity": "Mild (Benign)",
        "description": "HPV viral skin growth creating rough, raised, cauliflower-textured bumps with black pinpoint capillaries.",
        "red_flags": ["bleeding", "rapid spread in immunocompromised", "ulceration"]
    }
}

# Transmission Classification
CONTAGIOUS_MAP: Dict[str, str] = {
    # Contact (Contagious)
    "Chickenpox":           "Contact",
    "Monkeypox":            "Contact",
    "Ringworm":             "Contact",
    "warts":                "Contact",
    "tinea-versicolor":     "Contact",

    # Non-Contact (Non-contagious / Autoimmune / Inflammatory / Neoplastic)
    "Acne":                 "Non-Contact",
    "Pimple":               "Non-Contact",
    "Eczema":               "Non-Contact",
    "Psoriasis":            "Non-Contact",
    "vitiligo":             "Non-Contact",
    "basal cell carcinoma": "Non-Contact",
    "melanoma":             "Non-Contact"
}

# Critical Red Flag Emergency Keywords
RED_FLAG_KEYWORDS = [
    "fever", "high fever", "breathing", "shortness of breath", "swallowing",
    "lip swelling", "tongue swelling", "throat swelling", "anaphylaxis",
    "peeling skin", "skin sloughing", "skin peeling", "black skin", "necrosis",
    "unbearable pain", "extreme pain", "rapidly spreading", "red streaks",
    "confusion", "unconscious", "dizziness", "chest pain", "bleeding gums"
]

def check_red_flags(user_input: str) -> List[str]:
    """Scans user clinical notes for critical red flag symptoms."""
    if not user_input:
        return []
    user_input_lower = user_input.lower()
    detected_flags = []
    for flag in RED_FLAG_KEYWORDS:
        if flag in user_input_lower:
            detected_flags.append(flag)
    return detected_flags

def get_contagious_status(condition_key: str) -> str:
    """Returns 'Contact', 'Non-Contact', or 'Unknown'."""
    clean_key = condition_key.lower().replace("_", " ").strip()
    for k, v in CONTAGIOUS_MAP.items():
        if k.lower().replace("_", " ").strip() == clean_key:
            return v
    return "Non-Contact"

def get_condition_info(condition_key: str) -> Dict[str, Any]:
    """Fetch complete metadata record for a condition."""
    clean_key = condition_key.lower().replace("_", " ").strip()
    result = None
    for k, v in SYMPTOM_DB.items():
        if k.lower().replace("_", " ").strip() == clean_key:
            result = dict(v)
            break

    if result is None:
        result = {
            "symptoms": [],
            "severity": "Informational",
            "description": "Clinical skin lesion identified by YOLOv11 Edge AI Vision Engine.",
            "red_flags": []
        }

    result["contagious"] = get_contagious_status(condition_key)
    return result
