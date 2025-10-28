import torch
import joblib
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from torch.serialization import add_safe_globals

# -------------------------------------------------------------------
# Allow safe globals used in your saved .pt file
# -------------------------------------------------------------------
add_safe_globals([
    np._core.multiarray._reconstruct,
    np._core.multiarray.scalar,
    np.dtype,
    MultiLabelBinarizer
])

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
path_pt = "models/icd_cpt_distilbert_v3/label_binarizer.pt"
path_pkl = "models/icd_cpt_distilbert_v3/label_binarizer.pkl"

# -------------------------------------------------------------------
# Load safely
# -------------------------------------------------------------------
try:
    lb = torch.load(path_pt, map_location="cpu", weights_only=False)
    print("✅ Successfully loaded label_binarizer.pt")
except Exception as e:
    print(f"❌ Failed to load label_binarizer.pt: {e}")
    raise SystemExit

# -------------------------------------------------------------------
# Convert to pickle
# -------------------------------------------------------------------
joblib.dump(lb, path_pkl)
print(f"✅ Converted and saved label binarizer to {path_pkl}")

if hasattr(lb, "classes_"):
    print(f"Classes loaded: {len(lb.classes_)} → {lb.classes_[:10]}")
else:
    print("⚠️ Label binarizer loaded, but classes_ not found — verify file content.")
