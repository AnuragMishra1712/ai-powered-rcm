import os, re, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import torch
import torch.serialization
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast, DistilBertModel, get_linear_schedule_with_warmup

# ---------------- CONFIG ----------------
CSV_PATH = "doctor_notes_dataset_realistic_handwritten/data/doctor_notes_with_ocr.csv"
MODEL_OUT_DIR = "models/icd_cpt_distilbert_v3"
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"✅ Using device: {device}")

# ---------------- LOAD DATA ----------------
df = pd.read_csv(CSV_PATH).dropna(subset=["ocr_text"])

def parse_list(x):
    try: return eval(x)
    except: return [x] if isinstance(x,str) else []
df["icd10_codes"] = df["icd10_codes"].apply(parse_list)
df["cpt_codes"] = df["cpt_codes"].apply(parse_list)

def clean(t):
    t = str(t).lower()
    t = re.sub(r"[^a-z0-9\s.,-]", " ", t)
    return re.sub(r"\s+"," ",t).strip()
df["ocr_text"] = df["ocr_text"].apply(clean)

all_labels = sorted(list(set(sum(df["icd10_codes"],[])+sum(df["cpt_codes"],[]))))
mlb = MultiLabelBinarizer(classes=all_labels)
y = mlb.fit_transform(df["icd10_codes"]+df["cpt_codes"])

train_texts,val_texts,y_train,y_val = train_test_split(df["ocr_text"],y,test_size=0.2,random_state=42)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# ---------------- DATASET ----------------
class NotesDS(Dataset):
    def __init__(self,texts,labels,tokenizer,max_len=256):
        self.texts,self.labels,self.tok,self.max_len = list(texts),labels,tokenizer,max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self,idx):
        enc = self.tok(self.texts[idx],padding="max_length",truncation=True,max_length=self.max_len,return_tensors="pt")
        return {"input_ids":enc["input_ids"].squeeze(),
                "attention_mask":enc["attention_mask"].squeeze(),
                "labels":torch.tensor(self.labels[idx],dtype=torch.float)}

train_loader = DataLoader(NotesDS(train_texts,y_train,tokenizer),batch_size=8,shuffle=True)
val_loader   = DataLoader(NotesDS(val_texts,y_val,tokenizer),batch_size=8)

# ---------------- MODEL ----------------
class MultiLabelBERT(nn.Module):
    def __init__(self,num_labels):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        for p in list(self.bert.parameters())[:60]: p.requires_grad=False  # freeze first 3 layers
        self.drop = nn.Dropout(0.3)
        self.cls = nn.Linear(self.bert.config.hidden_size,num_labels)
    def forward(self,ids,mask):
        x = self.bert(input_ids=ids,attention_mask=mask).last_hidden_state[:,0]
        return self.cls(self.drop(x))

# Focal loss implementation
class FocalLoss(nn.Module):
    def __init__(self,alpha=1.0,gamma=2.0,reduction="mean"):
        super().__init__()
        self.alpha,self.gamma,self.reduction=alpha,gamma,reduction
    def forward(self,logits,targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits,targets,reduction="none")
        pt = torch.exp(-bce)
        loss = self.alpha*(1-pt)**self.gamma*bce
        return loss.mean() if self.reduction=="mean" else loss.sum()

num_labels = y.shape[1]
model = MultiLabelBERT(num_labels).to(device)
criterion = FocalLoss()
optimizer = AdamW(model.parameters(),lr=5e-5,weight_decay=0.01)
epochs=8
sched = get_linear_schedule_with_warmup(optimizer,0,len(train_loader)*epochs)

# ---------------- TRAIN ----------------
best_f1, patience, no_improve = 0, 3, 0
for epoch in range(epochs):
    model.train(); total_loss=0
    for b in tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs}"):
        ids,mask,labels = b["input_ids"].to(device),b["attention_mask"].to(device),b["labels"].to(device)
        optimizer.zero_grad()
        out = model(ids,mask)
        loss = criterion(out,labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step(); sched.step()
        total_loss+=loss.item()
    avg=total_loss/len(train_loader)

    # ---------- VAL ----------
    model.eval(); y_t,y_p=[],[]
    with torch.no_grad():
        for b in val_loader:
            ids,mask=b["input_ids"].to(device),b["attention_mask"].to(device)
            labels=b["labels"].cpu().numpy()
            preds=torch.sigmoid(model(ids,mask)).cpu().numpy()
            y_t.extend(labels); y_p.extend(preds)
    # dynamic threshold search
    best_t,best_f=0,0
    for t in np.arange(0.2,0.6,0.05):
        f=f1_score(y_t,(np.array(y_p)>t).astype(int),average="samples")
        if f>best_f: best_f,best_t=f,t
    print(f"\n🧩 Epoch {epoch+1} | Loss {avg:.4f} | BestF1 {best_f:.4f} @thr={best_t:.2f}")
    if best_f>best_f1:
        best_f1=best_f; no_improve=0
        torch.save({"model":model.state_dict(),"thr":best_t},os.path.join(MODEL_OUT_DIR,"best.pt"))
    else:
        no_improve+=1
        if no_improve>=patience: print("⏹️ Early stop."); break

# ---------------- EVAL ----------------
torch.serialization.add_safe_globals([np.core.multiarray.scalar])
ckpt = torch.load(os.path.join(MODEL_OUT_DIR, "best.pt"), weights_only=False)

model.load_state_dict(ckpt["model"])
thr = ckpt.get("thr", 0.3)
# print(f"\n⚙️ Loaded best checkpoint (thr={thr:.2f})")
print(f"\n⚙️ Loaded best checkpoint (thr={thr:.2f})")
model.eval(); y_t,y_p=[],[]
with torch.no_grad():
    for b in val_loader:
        ids,mask=b["input_ids"].to(device),b["attention_mask"].to(device)
        labels=b["labels"].cpu().numpy()
        preds=(torch.sigmoid(model(ids,mask)).cpu().numpy()>thr).astype(int)
        y_t.extend(labels); y_p.extend(preds)
print("\n📊 Final Validation Report:")
print(classification_report(y_t,y_p,target_names=mlb.classes_))
torch.save(mlb,os.path.join(MODEL_OUT_DIR,"label_binarizer.pt"))
tokenizer.save_pretrained(MODEL_OUT_DIR)
print(f"\n✅ Model saved in {MODEL_OUT_DIR}")
