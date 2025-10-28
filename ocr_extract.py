import os
import pandas as pd
from tqdm import tqdm
import pytesseract
from PIL import Image
import cv2
import numpy as np

# ---- CONFIG ----
DATA_CSV = "doctor_notes_dataset_realistic_handwritten/data/doctor_notes.csv"
OUTPUT_CSV = "doctor_notes_dataset_realistic_handwritten/data/doctor_notes_with_ocr.csv"

# ---- Load data ----
df = pd.read_csv(DATA_CSV)
ocr_texts = []

# ---- OCR Processing ----
for img_path in tqdm(df["image_path"], desc="Extracting text with OCR"):
    try:
        # Load image and convert to grayscale
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Preprocess to improve OCR
        gray = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.resize(thresh, None, fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC)
        
        # Convert back to PIL format for pytesseract
        pil_img = Image.fromarray(thresh)
        
        # Run OCR
        text = pytesseract.image_to_string(pil_img, lang="eng", config="--oem 3 --psm 6")
        text = text.replace("\n", " ").strip()
        ocr_texts.append(text)
    except Exception as e:
        print(f"⚠️ Error reading {img_path}: {e}")
        ocr_texts.append("")

# ---- Save results ----
df["ocr_text"] = ocr_texts
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ OCR extraction complete! Saved to: {OUTPUT_CSV}")
