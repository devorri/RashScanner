"""
symptoms_db.py - Clinical Symptom Database & Multimodal Matching Engine
Maps rash/skin conditions to clinical symptom criteria and calculates symptom match scores.
"""

import re
from typing import Dict, List, Set, Tuple, Any

# Clinical Symptom Database mapping condition keys to symptoms, severity, and red flag warnings
SYMPTOM_DB: Dict[str, Dict[str, Any]] = {
    "Eczema": {
        "symptoms": ["itchy", "dry", "red", "inflamed", "cracked", "rough", "scaling", "sensitive", "leathery"],
        "severity": "Mild to Moderate",
        "description": "Atopic Dermatitis causing dry, intensely itchy, inflamed, and scaling skin patches.",
        "red_flags": ["pus", "yellow crust", "fever", "severe pain", "spreading redness"]
    },
    "Hives": {
        "symptoms": ["itchy", "welts", "wheels", "red", "swollen", "raised", "stinging", "burning", "transient"],
        "severity": "Mild to Severe",
        "description": "Urticaria presenting as sudden, raised, itchy red or skin-colored welts.",
        "red_flags": ["difficulty breathing", "swallowing difficulty", "swollen lips", "swollen tongue", "anaphylaxis"]
    },
    "Ringworm": {
        "symptoms": ["circular", "ring-shaped", "itchy", "red", "scaly", "raised border", "clear center", "spreading"],
        "severity": "Mild",
        "description": "Tinea Corporis fungal infection producing ring-shaped scaly red patches.",
        "red_flags": ["spreading to face", "scalp involvement", "pus", "secondary bacterial infection"]
    },
    "Psoriasis": {
        "symptoms": ["silvery scales", "thickened", "red patches", "dry", "cracked", "itching", "burning", "plaque", "stiff joints"],
        "severity": "Moderate to Severe",
        "description": "Autoimmune condition marked by rapid skin cell buildup forming thick silvery scaly plaques.",
        "red_flags": ["joint swelling", "joint pain", "erythrodermic coverage over 90% body", "fever", "chills"]
    },
    "Contact_Dermatitis": {
        "symptoms": ["itchy", "red", "burning", "stinging", "blisters", "dry", "cracked", "swelling", "tender"],
        "severity": "Mild to Moderate",
        "description": "Inflammatory rash triggered by direct exposure to allergens or chemical irritants.",
        "red_flags": ["facial swelling", "eye involvement", "open weeping wounds", "severe blistering"]
    },
    "Acne_Vulgaris": {
        "symptoms": ["pimples", "blackheads", "whiteheads", "oily", "red bumps", "pustules", "nodules", "tender", "cysts"],
        "severity": "Mild to Moderate",
        "description": "Clogged hair follicles with oil and dead skin cells causing inflammatory lesions.",
        "red_flags": ["deep painful cysts", "scarring", "fever", "sudden explosive onset"]
    },
    "Actinic_keratosis": {
        "symptoms": ["rough", "scaly", "crusty", "dry", "sun-exposed", "sandpaper-feel", "pink", "brown", "flat to raised"],
        "severity": "Moderate (Pre-cancerous)",
        "description": "Pre-cancerous sun-damaged lesion with a rough, scaly patch on UV-exposed skin.",
        "red_flags": ["rapid growth", "bleeding", "ulceration", "induration", "pain"]
    },
    "Athlete_s_Foot_Tinea_Pedis": {
        "symptoms": ["itchy", "peeling", "cracked", "scaling", "between toes", "burning", "stinging", "blisters", "fissures"],
        "severity": "Mild",
        "description": "Tinea pedis fungal infection affecting the feet, soles, and interdigital spaces.",
        "red_flags": ["spreading to leg", "cellulitis", "fever", "pus"]
    },
    "Basal_cell_carcinoma": {
        "symptoms": ["pearly bump", "waxy", "translucent", "bleeding sore", "sun-exposed", "rolled border", "visible blood vessels"],
        "severity": "Severe (Malignancy)",
        "description": "Slow-growing non-melanoma skin cancer originating in basal cells.",
        "red_flags": ["non-healing ulcer", "recurrent bleeding", "rapid growth", "deep infiltration"]
    },
    "Candidiasis": {
        "symptoms": ["red rash", "itchy", "burning", "skin folds", "satellite lesions", "white patches", "moist", "macerated"],
        "severity": "Mild to Moderate",
        "description": "Yeast infection occurring in warm, moist skin folds (intertriginous areas).",
        "red_flags": ["fever", "chills", "disseminated spread", "immunosuppression"]
    },
    "Carbuncle": {
        "symptoms": ["painful cluster", "boils", "pus-filled", "red bump", "swollen", "deep infection", "draining", "tender"],
        "severity": "Moderate to Severe",
        "description": "Deep bacterial staphylococcus infection uniting multiple boils under the skin.",
        "red_flags": ["high fever", "chills", "rapid expansion", "confusion", "septic signs"]
    },
    "Cellulitis": {
        "symptoms": ["spreading redness", "warmth", "swelling", "pain", "tender", "tight skin", "red streaks", "fever"],
        "severity": "Severe (Emergency)",
        "description": "Potentially serious deep bacterial infection of the dermis and subcutaneous tissues.",
        "red_flags": ["high fever", "rapidly spreading redness", "red streaks towards heart", "confusion", "hypotension"]
    },
    "Cercarial_Dermatitis": {
        "symptoms": ["itchy", "red spots", "swimming", "tingling", "small papules", "blisters", "freshwater exposure"],
        "severity": "Mild",
        "description": "Swimmer's itch caused by allergic reaction to microscopic parasite larvae in water.",
        "red_flags": ["secondary bacterial infection", "pus", "persistent severe fever"]
    },
    "Cherry_Angioma": {
        "symptoms": ["bright red", "small dome", "smooth", "benign", "non-itchy", "cherry red", "painless"],
        "severity": "Mild (Benign)",
        "description": "Common harmless vascular skin growth consisting of a cluster of capillaries.",
        "red_flags": ["spontaneous heavy bleeding", "sudden change in shape or color"]
    },
    "Chickenpox_Varicella": {
        "symptoms": ["itchy blisters", "red spots", "dewdrops on rose petal", "fever", "fatigue", "crusting", "widespread"],
        "severity": "Moderate",
        "description": "Highly contagious varicella-zoster viral infection causing vesicular itchy rash.",
        "red_flags": ["difficulty breathing", "high fever", "neurological symptoms", "secondary skin infection"]
    },
    "Chromoblastomycosis": {
        "symptoms": ["warty nodules", "crusted papules", "cauliflower-like", "slow growing", "fungal", "extremities"],
        "severity": "Moderate to Severe",
        "description": "Chronic fungal subcutaneous infection producing verrucous warty lesions.",
        "red_flags": ["ulceration", "secondary bacterial superinfection", "limb swelling"]
    },
    "DRESS_Syndrome": {
        "symptoms": ["drug reaction", "widespread rash", "fever", "facial swelling", "swollen lymph nodes", "organ pain"],
        "severity": "Emergency",
        "description": "Severe systemic drug reaction with eosinophilia and systemic visceral symptoms.",
        "red_flags": ["high fever", "facial edema", "jaundice", "shortness of breath", "organ failure signs"]
    },
    "Dengue_Rash": {
        "symptoms": ["measles-like rash", "high fever", "severe body ache", "joint pain", "headache", "behind eyes pain", "flushing"],
        "severity": "Severe (Emergency)",
        "description": "Mosquito-borne viral dengue infection presenting with high fever and maculopapular exanthem.",
        "red_flags": ["bleeding gums", "nosebleeds", "persistent vomiting", "severe abdominal pain", "shock"]
    },
    "Dermatofibroma": {
        "symptoms": ["firm button bump", "brownish red", "dimple sign", "legs", "painless", "slow growing"],
        "severity": "Mild (Benign)",
        "description": "Harmless firm nodule beneath the skin, often forming a slight dimple when pinched.",
        "red_flags": ["rapid growth", "ulceration", "bleeding"]
    },
    "Dermatofibrosarcoma_Protuberans": {
        "symptoms": ["indurated plaque", "purplish red", "slow growing lump", "nodular", "firm depth"],
        "severity": "Severe (Malignancy)",
        "description": "Rare, locally invasive cutaneous soft tissue sarcoma.",
        "red_flags": ["rapid enlargement", "ulceration", "fixation to deep muscle"]
    },
    "Dermatomyositis": {
        "symptoms": ["heliotrope rash", "purple eyelids", "Gottron papules", "muscle weakness", "red knuckles", "fatigue"],
        "severity": "Severe",
        "description": "Inflammatory muscle disease characterized by muscle weakness and distinct violet skin rash.",
        "red_flags": ["difficulty swallowing", "shortness of breath", "severe muscle collapse"]
    },
    "Discoid_Lupus_Erythematosus": {
        "symptoms": ["coin-shaped", "red scaly plaques", "scarring", "sun-exposed", "hair loss", "pigment change", "plugged follicles"],
        "severity": "Moderate",
        "description": "Autoimmune condition causing chronic coin-shaped red scaly lesions that scar.",
        "red_flags": ["joint pain", "fever", "kidney signs", "systemic lupus flare"]
    },
    "Drug_eruption": {
        "symptoms": ["widespread red rash", "maculopapular", "symmetrical", "itchy", "starts after medication"],
        "severity": "Moderate to Severe",
        "description": "Adverse cutaneous reaction occurring days to weeks after initiating a drug.",
        "red_flags": ["skin blistering", "mucous membrane lesions", "fever", "skin sloughing"]
    },
    "Dyshidrotic_Eczema": {
        "symptoms": ["tiny deep blisters", "intensely itchy", "tapioca-like", "palms", "soles", "sides of fingers", "peeling"],
        "severity": "Moderate",
        "description": "Type of eczema causing tiny, itchy, fluid-filled vesicles on hands and feet.",
        "red_flags": ["pus", "spreading redness", "severe pain", "fever"]
    },
    "Ecthyma": {
        "symptoms": ["deep ulcer", "crusted sore", "pus", "punched-out appearance", "painful", "legs", "scarring"],
        "severity": "Moderate to Severe",
        "description": "Ulcerative form of impetigo extending deep into the dermis forming hard crusts.",
        "red_flags": ["fever", "cellulitis spread", "red streaks"]
    },
    "Epidermoid_Cyst": {
        "symptoms": ["slow growing lump", "central punctum", "cheesy discharge", "painless", "smooth dome"],
        "severity": "Mild",
        "description": "Benign sac under the skin filled with keratin material.",
        "red_flags": ["sudden rupture", "severe pain", "redness", "warmth", "foul pus"]
    },
    "Erysipelas": {
        "symptoms": ["bright red rash", "sharply demarcated border", "warmth", "swelling", "fever", "chills", "face", "legs"],
        "severity": "Severe (Emergency)",
        "description": "Acute superficial bacterial skin infection involving upper dermis and lymphatics.",
        "red_flags": ["high fever", "confusion", "rapid spreading", "blistering necrosis"]
    },
    "Erythrasma": {
        "symptoms": ["reddish-brown patches", "wrinkled", "skin folds", "groin", "armpits", "coral-red glow under UV"],
        "severity": "Mild",
        "description": "Superficial bacterial skin infection in skin folds caused by Corynebacterium minutissimum.",
        "red_flags": ["open skin cuts", "secondary bacterial infection"]
    },
    "Folliculitis": {
        "symptoms": ["small red bumps", "white-headed pimples", "around hair follicles", "itchy", "tender", "crusting"],
        "severity": "Mild",
        "description": "Inflammation or infection of hair follicles by bacteria or fungi.",
        "red_flags": ["large carbuncles", "fever", "spreading deep tissue infection"]
    },
    "Furunculosis": {
        "symptoms": ["boils", "painful red bumps", "pus head", "tender", "warmth", "recurrent boils"],
        "severity": "Moderate",
        "description": "Condition of having multiple or recurrent boils (furuncles) caused by staph.",
        "red_flags": ["fever", "boils on central face", "red streaks"]
    },
    "Granuloma_Annulare": {
        "symptoms": ["ring of bumps", "smooth papules", "reddish", "skin-colored", "hands", "feet", "non-itchy"],
        "severity": "Mild",
        "description": "Chronic benign skin condition presenting as circular smooth raised bumps.",
        "red_flags": ["widespread ulceration", "pain"]
    },
    "Hand_Foot_and_Mouth_Disease": {
        "symptoms": ["blisters in mouth", "red spots on hands", "red spots on feet", "fever", "sore throat", "loss of appetite"],
        "severity": "Mild to Moderate",
        "description": "Coxsackievirus viral infection producing characteristic vesicular sores on hands, feet, and mouth.",
        "red_flags": ["inability to swallow liquids", "dehydration", "high fever > 3 days", "lethargy"]
    },
    "Hansen_s_Disease": {
        "symptoms": ["numb skin patches", "lightened skin", "loss of sensation", "thickened nerves", "painless sores"],
        "severity": "Severe",
        "description": "Leprosy; chronic bacterial infection affecting peripheral nerves and skin.",
        "red_flags": ["loss of motor function", "unhealing ulcers", "eye damage"]
    },
    "Herpes_Gladiatorum": {
        "symptoms": ["clustered vesicles", "blisters", "face", "neck", "painful", "tingling", "athletes", "fever"],
        "severity": "Moderate",
        "description": "Herpes simplex viral infection common in contact sports athletes.",
        "red_flags": ["eye involvement", "lesions near cornea", "severe neurological pain"]
    },
    "Hidradenitis_Suppurativa": {
        "symptoms": ["painful deep lumps", "armpits", "groin", "tunnels", "foul discharge", "scarring", "recurrent"],
        "severity": "Moderate to Severe",
        "description": "Chronic inflammatory condition causing painful nodules and sinus tracts in apocrine gland areas.",
        "red_flags": ["severe systemic infection", "fever", "immobility due to pain"]
    },
    "Ichthyosis": {
        "symptoms": ["fish-like scales", "dry skin", "thickened", "flaking", "widespread", "cracked"],
        "severity": "Moderate",
        "description": "Genetic skin disorder causing persistent widespread dry, scaly, fish-like skin.",
        "red_flags": ["overheating", "inability to sweat", "severe deep skin infections"]
    },
    "Impetigo": {
        "symptoms": ["honey-colored crusts", "red sores", "blisters", "face", "around nose", "itchy", "contagious"],
        "severity": "Mild to Moderate",
        "description": "Highly contagious superficial bacterial skin infection forming classic golden honey crusts.",
        "red_flags": ["dark kidney-colored urine", "swelling around eyes (post-streptococcal kidney complication)", "fever"]
    },
    "Intertrigo": {
        "symptoms": ["red raw skin", "chafing", "skin folds", "under breasts", "groin", "stinging", "musty odor"],
        "severity": "Mild",
        "description": "Inflammatory skin condition caused by skin-on-skin friction, moisture, and heat.",
        "red_flags": ["fissures", "pus", "secondary fungal/bacterial infection"]
    },
    "Jock_Itch_Tinea_Cruris": {
        "symptoms": ["itchy groin", "red scaly border", "inner thighs", "spares scrotum", "burning", "chafing"],
        "severity": "Mild",
        "description": "Fungal infection of the groin and inner thighs.",
        "red_flags": ["spreading to genitals with severe pain", "pus"]
    },
    "Kaposi_Sarcoma": {
        "symptoms": ["purple macules", "dark reddish lesions", "plaques", "nodules", "legs", "mouth", "immunosuppression"],
        "severity": "Severe (Malignancy)",
        "description": "Cancer caused by HHV-8 creating purple or red vascular skin lesions.",
        "red_flags": ["gastrointestinal bleeding", "breathing difficulty", "rapid lymphatic swelling"]
    },
    "Keratosis_Pilaris": {
        "symptoms": ["goosebump feel", "tiny hard bumps", "upper arms", "thighs", "rough", "dry", "painless"],
        "severity": "Mild (Harmless)",
        "description": "Harmless keratin buildup plugging hair follicles producing a rough 'chicken skin' texture.",
        "red_flags": ["none (benign)"]
    },
    "Lichen_Planus": {
        "symptoms": ["purple papules", "pruritic", "polygonal", "planar", "shiny bumps", "white lacey lines", "wrists", "ankles"],
        "severity": "Moderate",
        "description": "Inflammatory condition producing itchy, flat-topped, polygonal violet bumps.",
        "red_flags": ["painful oral ulcers", "genital erosions", "hair follicle destruction"]
    },
    "Lichen_Simplex_Chronicus": {
        "symptoms": ["thickened skin", "leather-like", "intensely itchy", "hyperpigmented", "rubbing habit", "plaque"],
        "severity": "Moderate",
        "description": "Thickened, leathery skin resulting from repetitive scratching and rubbing.",
        "red_flags": ["secondary bacterial infection", "weeping pus"]
    },
    "Lipoma": {
        "symptoms": ["soft doughy lump", "movable under skin", "painless", "slow growing", "fatty tumor"],
        "severity": "Mild (Benign)",
        "description": "Common harmless benign fatty tissue tumor beneath the skin.",
        "red_flags": ["rapid growth", "firmness", "pain"]
    },
    "Lymphatic_filariasis": {
        "symptoms": ["severe leg swelling", "elephantiasis", "thickened rough skin", "lymph edema", "pain"],
        "severity": "Severe",
        "description": "Parasitic mosquito-borne infection causing lymphatic obstruction and massive tissue swelling.",
        "red_flags": ["acute bacterial lymphangitis", "high fever", "severe systemic infection"]
    },
    "Measles": {
        "symptoms": ["red maculopapular rash", "high fever", "cough", "runny nose", "coryza", "red eyes", "Koplik spots"],
        "severity": "Severe (Emergency)",
        "description": "Highly contagious viral infection causing classic high fever, cough, and spreading rash.",
        "red_flags": ["difficulty breathing", "stiff neck", "seizures", "confusion", "pneumonia signs"]
    },
    "Melanoma": {
        "symptoms": ["asymmetrical mole", "irregular border", "color variation", "diameter > 6mm", "evolving mole", "dark spot"],
        "severity": "Emergency (Malignancy)",
        "description": "Most dangerous form of skin cancer arising from pigment-producing melanocytes.",
        "red_flags": ["bleeding mole", "rapid evolution", "new black lesion", "itching or pain in mole"]
    },
    "Merkel_Cell_Carcinoma": {
        "symptoms": ["firm painless nodule", "red or purple bump", "rapidly growing", "sun-exposed skin", "elderly"],
        "severity": "Emergency (Malignancy)",
        "description": "Rare, aggressive neuroendocrine skin cancer.",
        "red_flags": ["rapid doubling in size", "ulceration", "lymph node enlargement"]
    },
    "Miliaria": {
        "symptoms": ["heat rash", "tiny clear bumps", "prickly heat", "red itchy bumps", "sweating", "chest", "back"],
        "severity": "Mild",
        "description": "Blocked sweat ducts causing sweat trapping under skin in hot, humid weather.",
        "red_flags": ["heat exhaustion signs", "fever", "pustules"]
    },
    "Molluscum_Contagiosum": {
        "symptoms": ["pearly dome bumps", "central dimple", "umbilicated", "small", "painless", "firm", "children"],
        "severity": "Mild",
        "description": "Viral skin infection producing small, firm, dome-shaped bumps with a central pit.",
        "red_flags": ["eye margin involvement", "widespread immunosuppressed outbreak"]
    },
    "Morphea_Localized_Scleroderma": {
        "symptoms": ["hardened skin patch", "waxy white center", "purple border", "tight skin", "loss of hair"],
        "severity": "Moderate",
        "description": "Localized autoimmune condition causing skin hardening and thickening.",
        "red_flags": ["joint contractures", "facial deformity", "deep muscle involvement"]
    },
    "Mycosis_Fungoides": {
        "symptoms": ["patch stage rash", "plaque stage", "red itchy patches", "tumors", "cutaneous T-cell lymphoma"],
        "severity": "Severe (Malignancy)",
        "description": "Most common form of cutaneous T-cell lymphoma presenting with chronic red eczema-like patches.",
        "red_flags": ["ulcerated tumors", "widespread erythroderma", "swollen lymph nodes"]
    },
    "Pemphigus": {
        "symptoms": ["fragile blisters", "flaccid blisters", "peeling skin", "painful mouth sores", "Nikolsky sign"],
        "severity": "Emergency",
        "description": "Rare autoimmune blistering disease causing fragile blisters on skin and mucous membranes.",
        "red_flags": ["widespread skin sloughing", "inability to eat/drink", "secondary sepsis"]
    },
    "Perioral_Dermatitis": {
        "symptoms": ["small red bumps around mouth", "spares lip border", "burning", "itching", "flaking", "around nose"],
        "severity": "Mild to Moderate",
        "description": "Facial rash around the mouth and nose often aggravated by topical steroids.",
        "red_flags": ["eye involvement", "severe painful pustules"]
    },
    "Pityriasis_Rosea": {
        "symptoms": ["herald patch", "christmas tree pattern", "salmon-pink oval patches", "scaly border", "mild itch"],
        "severity": "Mild",
        "description": "Benign self-limiting skin eruption starting with a single large herald patch.",
        "red_flags": ["severe systemic symptoms", "high fever"]
    },
    "Pityriasis_Versicolor": {
        "symptoms": ["discolored patches", "hypopigmented", "hyperpigmented", "fine scale", "chest", "back", "tinea versicolor"],
        "severity": "Mild",
        "description": "Fungal Malassezia yeast overgrowth causing lighter or darker scaly patches on trunk.",
        "red_flags": ["none"]
    },
    "Pyoderma_Gangrenosum": {
        "symptoms": ["rapidly enlarging ulcer", "violaceous border", "extremely painful", "purulent base", "pathergy"],
        "severity": "Emergency",
        "description": "Rare inflammatory disease resulting in large, painful deep ulcers.",
        "red_flags": ["rapid expansion in hours", "unbearable pain", "fever", "septic appearance"]
    },
    "Pyogenic_Granuloma": {
        "symptoms": ["red bleeding bump", "fleshy dome", "bleeds easily upon minor touch", "rapid growth", "fingers"],
        "severity": "Mild to Moderate",
        "description": "Benign vascular lesion that bleeds profusely upon minimal trauma.",
        "red_flags": ["uncontrollable arterial bleeding"]
    },
    "Rosacea": {
        "symptoms": ["facial redness", "flushing", "visible blood vessels", "telangiectasia", "red bumps", "rhinophyma", "sensitive"],
        "severity": "Moderate",
        "description": "Chronic inflammatory skin condition causing central facial redness and visible blood vessels.",
        "red_flags": ["ocular rosacea", "eye redness/pain", "vision changes"]
    },
    "Rubella": {
        "symptoms": ["pink rash", "starts on face", "fever", "swollen lymph nodes behind ears", "joint aches", "German measles"],
        "severity": "Moderate",
        "description": "Viral infection causing light pink rash and characteristic tender suboccipital lymph nodes.",
        "red_flags": ["pregnancy exposure (congenital rubella risk)", "high fever", "encephalitis signs"]
    },
    "Scabies": {
        "symptoms": ["intense night itching", "burrow lines", "webbing of fingers", "wrists", "waist", "small red papules"],
        "severity": "Moderate",
        "description": "Infestation by microscopic Sarcoptes scabiei mites causing severe nocturnal itch.",
        "red_flags": ["crusted scabies (Norwegian scabies)", "widespread secondary bacterial infection"]
    },
    "Schistosomiasis": {
        "symptoms": ["swimmer itch", "parasitic rash", "fever", "chills", "freshwater snail exposure", "Katayama fever"],
        "severity": "Moderate to Severe",
        "description": "Parasitic worm infection triggering acute cutaneous and systemic hypersensitivity.",
        "red_flags": ["bloody stool", "bloody urine", "high fever", "neurological symptoms"]
    },
    "Scrofuloderma": {
        "symptoms": ["tuberculous skin lesion", "cold abscess", "draining sinuses", "ulcerated skin over lymph node", "neck"],
        "severity": "Severe",
        "description": "Cutaneous tuberculosis extending from underlying infected lymph node or bone.",
        "red_flags": ["chronic cough", "weight loss", "night sweats", "systemic TB"]
    },
    "Sebaceous_Carcinoma": {
        "symptoms": ["painless firm yellow mass", "eyelid bump", "recurrent chalazion look", "elderly"],
        "severity": "Severe (Malignancy)",
        "description": "Uncommon aggressive cancer arising in sebaceous glands, frequently on eyelids.",
        "red_flags": ["eyelid loss", "rapid ulceration", "vision blockage"]
    },
    "Seborrheic_Dermatitis": {
        "symptoms": ["dandruff", "greasy yellow scales", "red scalp", "eyebrows", "sides of nose", "itchy flaking"],
        "severity": "Mild to Moderate",
        "description": "Common skin condition causing greasy scaly patches and red skin mainly on scalp and face.",
        "red_flags": ["widespread erythroderma", "severe pain", "pus"]
    },
    "Seborrheic_Keratosis": {
        "symptoms": ["waxy stuck-on growth", "brown black bump", "pasted-on appearance", "harmless", "wart-like"],
        "severity": "Mild (Benign)",
        "description": "Harmless, non-cancerous skin growth that appears waxy or glued onto the skin surface.",
        "red_flags": ["sudden eruptive appearance of hundreds of lesions (Sign of Leser-Trelat)"]
    },
    "Shingles_Herpes_Zoster": {
        "symptoms": ["painful band rash", "dermatomal", "blisters", "burning pain", "tingling", "one side of body", "fluid-filled"],
        "severity": "Severe",
        "description": "Reactivation of varicella-zoster virus causing painful localized band of vesicles along nerve path.",
        "red_flags": ["blisters near tip of nose or eye (Hutchinson sign)", "facial paralysis", "hearing loss"]
    },
    "Spider_Angioma": {
        "symptoms": ["central red spot", "spider-like leg vessels", "blanches with pressure", "face", "neck", "chest"],
        "severity": "Mild (Benign)",
        "description": "Small vascular dilation with central red punctum and radiating spider leg capillaries.",
        "red_flags": ["multiple sudden eruptive lesions (liver disease warning)"]
    },
    "Sporotrichosis": {
        "symptoms": ["rose gardener disease", "nodular lymph chain", "painless bump", "ulcerating bump", "thorns exposure"],
        "severity": "Moderate to Severe",
        "description": "Fungal infection acquired through plant thorn pricks, spreading along lymphatic channels.",
        "red_flags": ["disseminated systemic fungal spread", "joint swelling"]
    },
    "Squamous_cell_carcinoma": {
        "symptoms": ["firm red nodule", "crusty flat sore", "non-healing ulcer", "rough scaly patch", "sun-exposed"],
        "severity": "Severe (Malignancy)",
        "description": "Second most common skin cancer arising in squamous cells of outer epidermis.",
        "red_flags": ["rapid expansion", "deep tenderness", "lymph node swelling"]
    },
    "Stasis_Dermatitis": {
        "symptoms": ["lower leg swelling", "reddish-brown discoloration", "hemosiderin staining", "itchy", "varicose veins", "scaly"],
        "severity": "Moderate",
        "description": "Skin inflammation on lower legs resulting from poor blood circulation and venous insufficiency.",
        "red_flags": ["deep venous thrombosis (DVT) swelling", "venous leg ulceration", "cellulitis"]
    },
    "Stevens_Johnson_Syndrome": {
        "symptoms": ["blistering rash", "skin sloughing", "lip ulceration", "eye inflammation", "fever", "medication reaction"],
        "severity": "Emergency",
        "description": "Rare, severe life-threatening reaction affecting skin and mucous membranes.",
        "red_flags": ["skin peeling off in sheets", "mouth ulcers", "eye damage", "high fever"]
    },
    "Toxic_Epidermal_Necrolysis": {
        "symptoms": ["widespread skin peeling > 30%", "scalded skin feel", "blistering", "fever", "mucous membrane detachment"],
        "severity": "Emergency",
        "description": "Most severe form of Stevens-Johnson spectrum with extensive epidermal detachment.",
        "red_flags": ["skin sloughing off over 30% body area", "sepsis", "respiratory distress"]
    },
    "Warts": {
        "symptoms": ["rough bump", "verrucous surface", "black pinpoint dots", "hands", "feet", "plantar", "verruca"],
        "severity": "Mild",
        "description": "HPV viral skin growth creating rough, raised bumps on skin.",
        "red_flags": ["genital area involvement", "bleeding", "rapid spread in immunocompromised"]
    },
    "Yaws": {
        "symptoms": ["raspberry-like bump", "mother yaw", "painless ulcer", "joint pain", "tropical bacterial infection"],
        "severity": "Moderate",
        "description": "Chronic tropical infection caused by Treponema pertenue producing raspberry-like skin papules.",
        "red_flags": ["bone deformities", "facial destructive lesions"]
    }
}

# Red Flag Symptom Keywords triggering safety alerts
RED_FLAG_KEYWORDS = [
    "fever", "high fever", "breathing", "shortness of breath", "swallowing", 
    "lip swelling", "tongue swelling", "throat swelling", "anaphylaxis",
    "peeling skin", "skin sloughing", "skin peeling", "black skin", "necrosis",
    "unbearable pain", "extreme pain", "rapidly spreading", "red streaks",
    "confusion", "unconscious", "dizziness", "chest pain", "bleeding gums"
]


def normalize_text(text: str) -> List[str]:
    """Convert input string to normalized list of lower-case symptom tokens."""
    if isinstance(text, list):
        text = " ".join(text)
    text = text.lower()
    # Remove punctuation and split words
    words = re.findall(r'\b[a-z0-9_\-]+\b', text)
    return words


def check_red_flags(user_input: str) -> List[str]:
    """
    Scans user input for critical red flag symptoms.
    Returns list of matched safety warnings if found.
    """
    user_input_lower = user_input.lower()
    detected_flags = []
    
    for flag in RED_FLAG_KEYWORDS:
        if flag in user_input_lower:
            detected_flags.append(flag)
            
    return detected_flags


def calculate_symptom_score(user_input: str, condition_key: str) -> float:
    """
    Calculates a normalized symptom match score S_score in [0.0, 1.0] for a given condition.
    
    Formula:
        Matching Symptom Count / Total Required Symptoms for Condition
        Weighted with Jaccard similarity to reward specific matches.
    """
    if condition_key not in SYMPTOM_DB:
        # Check normalized match key
        clean_key = condition_key.replace(" ", "_")
        found = None
        for k in SYMPTOM_DB:
            if k.lower() == condition_key.lower() or k.lower() == clean_key.lower():
                found = k
                break
        if found:
            condition_key = found
        else:
            return 0.0

    condition_info = SYMPTOM_DB[condition_key]
    db_symptoms = [s.lower() for s in condition_info["symptoms"]]
    
    user_tokens = set(normalize_text(user_input))
    if not user_tokens:
        return 0.0

    matched_count = 0
    for sym in db_symptoms:
        sym_tokens = set(normalize_text(sym))
        # Match if any token in multi-word symptom exists in user input or exact match
        if sym_tokens.issubset(user_tokens) or any(t in user_tokens for t in sym_tokens):
            matched_count += 1

    if not db_symptoms:
        return 0.0

    # Recall ratio: how many of condition's key symptoms matched
    recall = matched_count / len(db_symptoms)
    
    # Simple coverage ratio
    return min(1.0, recall * 1.2)  # Mild boost factor for high matching symptoms


def get_all_symptoms() -> List[str]:
    """Return a unique sorted list of all available symptom keywords across the DB."""
    all_syms = set()
    for cond_info in SYMPTOM_DB.values():
        for s in cond_info["symptoms"]:
            all_syms.add(s)
    return sorted(list(all_syms))


# =============================================================================
# Contact / Non-Contact Classification for all 75 conditions
# "Contact"     = Contagious — can spread from person to person or via surfaces
# "Non-Contact" = Not contagious — autoimmune, genetic, environmental, neoplastic
# =============================================================================
CONTAGIOUS_MAP: Dict[str, str] = {
    # CONTACT (Contagious)
    "Ringworm":                        "Contact",
    "Athlete_s_Foot_Tinea_Pedis":      "Contact",
    "Jock_Itch_Tinea_Cruris":          "Contact",
    "Tinea_Corporis":                  "Contact",
    "Candidiasis":                     "Contact",
    "Impetigo":                        "Contact",
    "Ecthyma":                         "Contact",
    "Scabies":                         "Contact",
    "Warts":                           "Contact",
    "Molluscum_Contagiosum":           "Contact",
    "Chickenpox_Varicella":            "Contact",
    "Shingles_Herpes_Zoster":         "Contact",
    "Herpes_Gladiatorum":              "Contact",
    "Measles":                         "Contact",
    "Rubella":                         "Contact",
    "Hand_Foot_and_Mouth_Disease":     "Contact",
    "Hansen_s_Disease":                "Contact",
    "Folliculitis":                    "Contact",
    "Furunculosis":                    "Contact",
    "Carbuncle":                       "Contact",
    "Cellulitis":                      "Contact",
    "Erysipelas":                      "Contact",
    "Erythrasma":                      "Contact",
    "Sporotrichosis":                  "Contact",
    "Chromoblastomycosis":             "Contact",
    "Scrofuloderma":                   "Contact",
    "Yaws":                            "Contact",
    "Lymphatic_filariasis":            "Contact",
    "Schistosomiasis":                 "Contact",
    "Cercarial_Dermatitis":            "Contact",
    "Dengue_Rash":                     "Contact",
    "Pityriasis_Versicolor":           "Contact",
    "Intertrigo":                      "Contact",
    "Suborrheic_Dermatitis":           "Contact",

    # NON-CONTACT (Not Contagious)
    "Eczema":                          "Non-Contact",
    "Hives":                           "Non-Contact",
    "Psoriasis":                       "Non-Contact",
    "Contact_Dermatitis":              "Non-Contact",
    "Acne_Vulgaris":                   "Non-Contact",
    "Rosacea":                         "Non-Contact",
    "Seborrheic_Dermatitis":           "Non-Contact",
    "Seborrheic_Keratosis":            "Non-Contact",
    "Keratosis_Pilaris":               "Non-Contact",
    "Actinic_keratosis":               "Non-Contact",
    "Melanoma":                        "Non-Contact",
    "Basal_cell_carcinoma":            "Non-Contact",
    "Squamous_cell_carcinoma":         "Non-Contact",
    "Merkel_Cell_Carcinoma":           "Non-Contact",
    "Kaposi_Sarcoma":                  "Non-Contact",
    "Dermatofibrosarcoma_Protuberans": "Non-Contact",
    "Mycosis_Fungoides":               "Non-Contact",
    "Sebaceous_Carcinoma":             "Non-Contact",
    "Discoid_Lupus_Erythematosus":     "Non-Contact",
    "Dermatomyositis":                 "Non-Contact",
    "Pemphigus":                       "Non-Contact",
    "Stevens_Johnson_Syndrome":        "Non-Contact",
    "Toxic_Epidermal_Necrolysis":      "Non-Contact",
    "DRESS_Syndrome":                  "Non-Contact",
    "Drug_eruption":                   "Non-Contact",
    "Dyshidrotic_Eczema":              "Non-Contact",
    "Lichen_Planus":                   "Non-Contact",
    "Lichen_Simplex_Chronicus":        "Non-Contact",
    "Granuloma_Annulare":              "Non-Contact",
    "Morphea_Localized_Scleroderma":  "Non-Contact",
    "Ichthyosis":                      "Non-Contact",
    "Pityriasis_Rosea":                "Non-Contact",
    "Miliaria":                        "Non-Contact",
    "Stasis_Dermatitis":               "Non-Contact",
    "Pyoderma_Gangrenosum":            "Non-Contact",
    "Hidradenitis_Suppurativa":        "Non-Contact",
    "Perioral_Dermatitis":             "Non-Contact",
    "Dermatofibroma":                  "Non-Contact",
    "Lipoma":                          "Non-Contact",
    "Epidermoid_Cyst":                 "Non-Contact",
    "Cherry_Angioma":                  "Non-Contact",
    "Spider_Angioma":                  "Non-Contact",
    "Pyogenic_Granuloma":              "Non-Contact",
    "Sweets_Syndrome":                 "Non-Contact",
    "Sweets_Syndrome_acute_febrile_neutrophilic_dermatosis": "Non-Contact",
    "Mycetoma":                        "Non-Contact",
}


def get_contagious_status(condition_key: str) -> str:
    """Returns 'Contact', 'Non-Contact', or 'Unknown' for a given condition."""
    if condition_key in CONTAGIOUS_MAP:
        return CONTAGIOUS_MAP[condition_key]
    # Fuzzy fallback — normalize underscores/spaces
    clean_key = condition_key.replace(" ", "_")
    for k, v in CONTAGIOUS_MAP.items():
        if k.lower() == condition_key.lower() or k.lower() == clean_key.lower():
            return v
    return "Unknown"


def get_condition_info(condition_key: str) -> Dict[str, Any]:
    """Fetch complete metadata record for a condition, including contagious status."""
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
            "severity": "Unknown",
            "description": "No detailed clinical description available.",
            "red_flags": []
        }

    # Attach contagious classification
    result["contagious"] = get_contagious_status(condition_key)
    return result


if __name__ == "__main__":
    print(f"[Info] Symptom Database loaded with {len(SYMPTOM_DB)} conditions.")
    print(f"[Info] Contagious map loaded with {len(CONTAGIOUS_MAP)} entries.")
    sample_input = "itchy dry red skin cracked scaling"
    print(f"\nTesting sample user input: '{sample_input}'")

    for cond in ["Eczema", "Psoriasis", "Ringworm", "Scabies", "Cellulitis"]:
        score = calculate_symptom_score(sample_input, cond)
        contagious = get_contagious_status(cond)
        print(f"  - {cond} [{contagious}]: {score*100:.1f}%")
