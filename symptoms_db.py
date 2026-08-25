"""
symptoms_db.py - Clinical Information & Safety Database for 10 Skin Conditions
Provides clinical descriptions, transmission classification (Contact vs Non-Contact),
and critical emergency Red-Flag safety checks.
Note: AI diagnosis is 100% Computer Vision-driven. Symptoms in this module are used for
clinician reference notes and red-flag emergency detection.
"""

import re
from typing import Dict, List, Any

# Clinical Reference Information for the 10 Selected Visually-Distinct Conditions
SYMPTOM_DB: Dict[str, Dict[str, Any]] = {
    "Acne_Vulgaris": {
        "symptoms": ["pimples", "blackheads", "whiteheads", "oily", "red bumps", "pustules", "nodules", "tender", "cysts"],
        "severity": "Mild to Moderate",
        "description": "Clogged hair follicles with oil and dead skin cells causing inflammatory papules, pustules, or cysts.",
        "red_flags": ["deep painful cysts", "scarring", "fever", "sudden explosive onset"]
    },
    "Chickenpox_Varicella": {
        "symptoms": ["itchy blisters", "red spots", "dewdrops on rose petal", "fever", "fatigue", "crusting", "widespread"],
        "severity": "Moderate",
        "description": "Highly contagious varicella-zoster viral infection causing itchy fluid-filled vesicles and crusts.",
        "red_flags": ["difficulty breathing", "high fever", "neurological symptoms", "secondary skin infection"]
    },
    "Eczema": {
        "symptoms": ["itchy", "dry", "red", "inflamed", "cracked", "rough", "scaling", "sensitive", "leathery"],
        "severity": "Mild to Moderate",
        "description": "Atopic Dermatitis causing dry, intensely itchy, inflamed, and scaling skin patches.",
        "red_flags": ["pus", "yellow crust", "fever", "severe pain", "spreading redness"]
    },
    "Hives": {
        "symptoms": ["itchy", "welts", "wheals", "red", "swollen", "raised", "stinging", "burning", "transient"],
        "severity": "Mild to Severe",
        "description": "Urticaria presenting as sudden, raised, itchy red or pink welts with smooth surface.",
        "red_flags": ["difficulty breathing", "swallowing difficulty", "swollen lips", "swollen tongue", "anaphylaxis"]
    },
    "Impetigo": {
        "symptoms": ["honey-colored crusts", "red sores", "blisters", "face", "around nose", "itchy", "contagious"],
        "severity": "Mild to Moderate",
        "description": "Contagious superficial bacterial skin infection forming characteristic golden honey-colored crusts.",
        "red_flags": ["dark kidney-colored urine", "swelling around eyes", "high fever"]
    },
    "Melanoma": {
        "symptoms": ["asymmetrical mole", "irregular border", "color variation", "diameter > 6mm", "evolving mole", "dark spot"],
        "severity": "Emergency (Malignancy)",
        "description": "High-risk skin malignancy arising from melanocytes showing asymmetry, irregular borders, and color variation.",
        "red_flags": ["bleeding mole", "rapid evolution", "new black lesion", "itching or pain in mole"]
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
    "Scabies": {
        "symptoms": ["intense night itching", "burrow lines", "webbing of fingers", "wrists", "waist", "small red papules"],
        "severity": "Moderate",
        "description": "Infestation by microscopic Sarcoptes scabiei mites causing intense nocturnal itch and burrow tracks.",
        "red_flags": ["crusted scabies (Norwegian scabies)", "widespread secondary bacterial infection"]
    },
    "Warts": {
        "symptoms": ["rough bump", "verrucous surface", "black pinpoint dots", "hands", "feet", "plantar", "verruca"],
        "severity": "Mild (Benign)",
        "description": "HPV viral skin growth creating rough, raised, cauliflower-textured bumps with black pinpoint capillaries.",
        "red_flags": ["bleeding", "rapid spread in immunocompromised", "ulceration"]
    }
}

# Transmission Classification
CONTAGIOUS_MAP: Dict[str, str] = {
    # Contact (Contagious)
    "Chickenpox_Varicella": "Contact",
    "Impetigo":             "Contact",
    "Ringworm":             "Contact",
    "Scabies":              "Contact",
    "Warts":                "Contact",

    # Non-Contact (Non-contagious / Autoimmune / Inflammatory / Neoplastic)
    "Acne_Vulgaris":        "Non-Contact",
    "Eczema":               "Non-Contact",
    "Hives":                "Non-Contact",
    "Melanoma":             "Non-Contact",
    "Psoriasis":            "Non-Contact"
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
    if condition_key in CONTAGIOUS_MAP:
        return CONTAGIOUS_MAP[condition_key]
    clean_key = condition_key.replace(" ", "_")
    for k, v in CONTAGIOUS_MAP.items():
        if k.lower() == condition_key.lower() or k.lower() == clean_key.lower():
            return v
    return "Non-Contact"

def get_condition_info(condition_key: str) -> Dict[str, Any]:
    """Fetch complete metadata record for a condition."""
    result = None
    if condition_key in SYMPTOM_DB:
        result = dict(SYMPTOM_DB[condition_key])
    else:
        clean_key = condition_key.replace(" ", "_")
        for k, v in SYMPTOM_DB.items():
            if k.lower() == condition_key.lower() or k.lower() == clean_key.lower():
                result = dict(v)
                break

    if result is None:
        result = {
            "symptoms": [],
            "severity": "Informational",
            "description": "Clinical condition identified by Edge AI Vision Engine.",
            "red_flags": []
        }

    result["contagious"] = get_contagious_status(condition_key)
    return result
