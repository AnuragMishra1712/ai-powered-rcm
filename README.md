Perfect — let’s add both ✅ a **`requirements.txt`** and ✅ a **professional `README.md`** inside your `ai_rcm_project/` folder so anyone can easily install and run your app on Windows or macOS.

Below are **ready-to-copy full files** 👇

---

## 📄 `ai_rcm_project/requirements.txt`

```txt
# Core dependencies
streamlit==1.39.0
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2

# Machine Learning Models
catboost==1.2.5
xgboost==2.1.1
lightgbm==4.3.0

# Utility / Formatting
scikit-learn==1.5.2
matplotlib==3.9.2
regex==2024.9.11

# Optional for deployment and performance
pyarrow>=15.0.0
```

---

## 📘 `ai_rcm_project/README.md`

````markdown
# AI-Powered Revenue Cycle Management (RCM) Suite

## 🧠 Overview
This project is a **Streamlit-based AI web application** that automates and optimizes the healthcare revenue cycle — from claim submission to collections — using Machine Learning.

The app includes:
1. **Denial Prediction & Prevention** (CatBoost)
2. **AI-Assisted Coding** (Pattern-based NLP)
3. **Prior Authorization Automation** (LightGBM)
4. **Billing & Collections Optimization** (XGBoost)

Each module predicts, simulates, or automates part of the revenue workflow — helping reduce denials, improve coding accuracy, and speed up collections.

---

## ⚙️ Tech Stack
- **Frontend:** Streamlit
- **Backend Models:** CatBoost, LightGBM, XGBoost
- **Language:** Python 3.11
- **Environment:** venv (recommended)
- **Data:** Synthetic 50K-row simulated healthcare dataset

---

## 💻 Installation (Windows / macOS)

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AnuragMishra1712/ai-powered-rcm.git
cd ai-powered-rcm/ai_rcm_project
````

### 2️⃣ Create Virtual Environment

**Windows (CMD or PowerShell):**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app/app.py
```

If Streamlit opens a URL like `http://localhost:8501/`, click it or paste it into your browser.

---

## 🧩 Folder Structure

```
ai_rcm_project/
│
├── app/
│   └── app.py                  # Main Streamlit app
│
├── models/
│   ├── denial_model.cbm
│   ├── pa_model.txt
│   ├── billing_model.json
│   └── coding_model.pkl
│
├── data/                       # (Optional) Synthetic datasets
│
├── requirements.txt
└── README.md
```

---

## 🧪 Notes for New Users

* The models are **pre-trained on generated healthcare data** (safe, synthetic).
* You do **not** need Ollama or any external LLMs — all inference is local.
* Make sure **`models/`** folder exists inside the same root as `app/`.
* If you’re using Windows, run all commands from inside the virtual environment.

---

## 🚀 Key Features at a Glance

* Predict claim denials before they occur.
* Auto-generate CPT/ICD-10 codes from clinical notes.
* Simulate prior-authorization workflow using ML.
* Predict payment likelihood and automate billing actions.

---

## 🧾 Author

**Anurag Mishra**
Data & AI Engineer — Jersey City, NJ
📧 [mishra.anurag1712@gmail.com](mailto:mishra.anurag1712@gmail.com)
🔗 [linkedin.com/in/amanuragmishra](https://linkedin.com/in/amanuragmishra)

---

## ⚖️ License

This project is for **educational and demonstration purposes only.**
It uses synthetic data and does not access any real patient information.

```

---

### ✅ Next Steps
1. Copy both files into your folder:
```

ai_rcm_project/requirements.txt
ai_rcm_project/README.md

````
2. Commit and push again:
```bash
git add ai_rcm_project/requirements.txt ai_rcm_project/README.md
git commit -m "Added requirements and README"
git push
````


