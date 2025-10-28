import os, random, json
from faker import Faker
import pandas as pd
from tqdm import tqdm
from handright import Template, handwrite
from PIL import Image, ImageFont

fake = Faker()

# ---------------- CONFIG ----------------
N_SAMPLES = 500
SPECIALTIES = ["Cardiology", "Orthopedics", "Neurology", "Pediatrics", "General Medicine"]
OUT_DIR = "doctor_notes_dataset_realistic_handwritten"
IMG_DIR = os.path.join(OUT_DIR, "data", "handwritten_images")
os.makedirs(IMG_DIR, exist_ok=True)

ICD10_CODES = {
    "Cardiology": {"I10": "Hypertension", "I21.3": "Acute MI", "I25.10": "CAD", "R07.9": "Chest pain"},
    "Orthopedics": {"M17.9": "Knee osteoarthritis", "S83.241A": "ACL tear", "M54.5": "Low back pain"},
    "Neurology": {"G40.909": "Epilepsy", "G43.909": "Migraine", "R51": "Headache"},
    "Pediatrics": {"J45.909": "Asthma", "A09": "Gastroenteritis", "H66.90": "Otitis media"},
    "General Medicine": {"E11.9": "Type 2 diabetes", "J06.9": "URI", "R50.9": "Fever"}
}

CPT_CODES = {
    "Cardiology": {"93000": "ECG", "99213": "Consultation", "99214": "Follow-up"},
    "Orthopedics": {"73562": "Knee X-ray", "20610": "Joint injection", "99213": "Consultation"},
    "Neurology": {"95816": "EEG", "99214": "Follow-up", "96116": "Neuro eval"},
    "Pediatrics": {"99391": "Well child exam", "99213": "Consult", "87880": "Strep test"},
    "General Medicine": {"99213": "Office visit", "81002": "Urine test", "80053": "CMP panel"}
}

# -------------- Note generator --------------
def random_note_text(specialty):
    patient = fake.name()
    age = random.randint(18, 80)
    symptoms = {
        "Cardiology": ["chest pain", "shortness of breath", "palpitations"],
        "Orthopedics": ["knee pain", "shoulder stiffness", "lower back pain"],
        "Neurology": ["headache", "seizure", "numbness"],
        "Pediatrics": ["fever", "cough", "ear pain"],
        "General Medicine": ["fatigue", "fever", "body ache"]
    }[specialty]
    treatment = {
        "Cardiology": ["started on aspirin", "beta-blocker initiated", "advised ECG and cath lab"],
        "Orthopedics": ["NSAIDs prescribed", "advised physiotherapy", "MRI recommended"],
        "Neurology": ["EEG planned", "anti-seizure medication started"],
        "Pediatrics": ["paracetamol advised", "hydration and rest"],
        "General Medicine": ["blood tests ordered", "fluids advised", "antibiotics if no improvement"]
    }[specialty]

    icd_choices = random.sample(list(ICD10_CODES[specialty].items()), k=random.randint(1, 2))
    cpt_choices = random.sample(list(CPT_CODES[specialty].items()), k=random.randint(1, 2))

    icd_text = ", ".join([f"{code} ({desc})" for code, desc in icd_choices])
    cpt_text = ", ".join([f"{code} ({desc})" for code, desc in cpt_choices])

    txt = (
        f"Patient: {patient}, Age: {age}.\n"
        f"Chief complaint: {random.choice(symptoms)} for {random.randint(2,10)} days.\n"
        f"Diagnosis: {icd_text}\n"
        f"CPT: {cpt_text}\n"
        f"Treatment: {random.choice(treatment)}.\n"
        f"Follow-up in {random.randint(3,14)} days."
    )

    icd_codes = [c for c, _ in icd_choices]
    cpt_codes = [c for c, _ in cpt_choices]
    return txt, icd_codes, cpt_codes

# -------------- Handwriting template --------------
font_path = "/Library/Fonts/JustAnotherHand-Regular.ttf"
template = Template(
    background=Image.new("RGB", (1280, 720), (245, 245, 240)),
    font=ImageFont.truetype(font_path, 46),
    line_spacing=85,
    fill=0,
    left_margin=100,
    top_margin=100,
    right_margin=100,
    bottom_margin=100,
    word_spacing_sigma=5,
    line_spacing_sigma=4,
    font_size_sigma=2,
)

# -------------- Generate images --------------
records = []
for i in tqdm(range(1, N_SAMPLES + 1), desc="Generating realistic notes"):
    note_id = f"NOTE_{i:04d}"
    specialty = random.choice(SPECIALTIES)
    note_text, icd10, cpt = random_note_text(specialty)

    images = list(handwrite(note_text, template))
    image_path = os.path.join(IMG_DIR, f"{note_id}.png")
    images[0].save(image_path, optimize=True, quality=85)

    records.append({
        "note_id": note_id,
        "specialty": specialty,
        "note_text": note_text,
        "icd10_codes": icd10,
        "cpt_codes": cpt,
        "image_path": image_path
    })

# -------------- Save metadata --------------
os.makedirs(os.path.join(OUT_DIR, "data"), exist_ok=True)
pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, "data", "doctor_notes.csv"), index=False)
with open(os.path.join(OUT_DIR, "data", "doctor_notes.json"), "w") as f:
    json.dump(records, f, indent=2)

print("\n✅ Generated realistic handwritten dataset (ICD + CPT visible) at:", OUT_DIR)
