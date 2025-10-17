import pandas as pd
import random
import os

# Ensure reproducibility
random.seed(42)

os.makedirs("data", exist_ok=True)

# ----------------------------
# Reference pools (realistic medical relationships)
# ----------------------------
conditions = [
    {"symptom": "chest pain", "specialty": "Cardiology", "cpt": ["93000", "93015"], "icd": ["R07.9", "I20.9"]},
    {"symptom": "abdominal pain", "specialty": "Gastroenterology", "cpt": ["45378", "99214"], "icd": ["K62.5", "R10.9"]},
    {"symptom": "joint pain", "specialty": "Orthopedics", "cpt": ["73560", "99213"], "icd": ["M25.50"]},
    {"symptom": "shortness of breath", "specialty": "Pulmonology", "cpt": ["71250", "99284"], "icd": ["R06.02"]},
    {"symptom": "diabetes follow-up", "specialty": "Endocrinology", "cpt": ["99214"], "icd": ["E11.9"]},
    {"symptom": "skin rash", "specialty": "Dermatology", "cpt": ["11102", "99213"], "icd": ["L98.9"]},
    {"symptom": "headache", "specialty": "Neurology", "cpt": ["70450", "99214"], "icd": ["R51"]},
    {"symptom": "preventive exam", "specialty": "Family Medicine", "cpt": ["99396", "88142"], "icd": ["Z00.00"]}
]

orders_pool = [
    "cbc", "lipid panel", "flu vaccine", "chest xray", "ct head", "ekg",
    "pap smear", "colonoscopy biopsy", "mri knee", "stress test", "ct chest"
]

procedure_reports = [
    "Procedure completed successfully with no complications.",
    "Findings consistent with patient’s clinical symptoms.",
    "Normal results observed during imaging and diagnostic testing.",
    "Abnormal findings noted — further evaluation recommended.",
    "Samples sent for pathology; results pending."
]

# ----------------------------
# Data generation
# ----------------------------
records = []
for i in range(1, 50001):
    encounter_id = f"E{i:05d}"
    provider_id = f"P{random.randint(100, 999)}"
    coder_id = f"C{random.randint(100, 120)}"

    case = random.choice(conditions)
    symptom = case["symptom"]
    specialty = case["specialty"]

    # Random note assembly
    order_list = random.sample(orders_pool, random.randint(1, 3))
    procedure = random.choice(case["cpt"])
    icd = random.choice(case["icd"])

    note_text = (
        f"Patient presents with {symptom}. "
        f"{specialty} evaluation performed. "
        f"Orders placed: {', '.join(order_list)}. "
        f"Procedure: {procedure} performed. "
        "No adverse events noted. Follow-up in 2 weeks."
    )

    procedure_report = random.choice(procedure_reports)

    # Time and metadata
    time_to_code = random.randint(10, 60)
    visit_month = random.randint(1, 12)
    state = random.choice(["NJ", "CA", "TX", "NY", "FL", "IL"])
    insurance_type = random.choice(["Private", "Medicare", "Medicaid", "Self-pay"])
    employment_status = random.choice(["Employed", "Unemployed", "Retired"])
    chronic_condition = random.choice([True, False])

    record = {
        "encounter_id": encounter_id,
        "provider_id": provider_id,
        "note_text": note_text,
        "orders": str(order_list),
        "procedure_reports": procedure_report,
        "final_billed_cpt": str([procedure]),
        "final_icd10": str([icd]),
        "time_to_code": time_to_code,
        "coder_id": coder_id,
        "specialty": specialty,
        "insurance_type": insurance_type,
        "state": state,
        "employment_status": employment_status,
        "visit_month": visit_month,
        "chronic_condition": chronic_condition
    }

    records.append(record)

# ----------------------------
# Save dataset
# ----------------------------
df = pd.DataFrame(records)
output_path = "data/coding_data.csv"
df.to_csv(output_path, index=False)

print(f"✅ Generated {len(df)} rows with {df.shape[1]} columns")
print(f"💾 Saved to {output_path}\n")
print("🔍 Sample:")
print(df.head(3))

