# 🔍 Credit Card Fraud Detection
## 📌 Business Problem
Banks and fintech companies lose billions every year to fraudulent transactions. This project builds an end-to-end machine learning pipeline that detects fraud and assigns every transaction a real-time anomaly risk score.
---
## 📁 Dataset
**Source:** [Kaggle — Credit Card Fraud Detection by MLG-ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
| File | Description |
|---|---|
| creditcard.csv | 284,807 transactions · 30 features · labeled fraud or legit |
> ⚠️ Not included in this repo (143 MB). Download from Kaggle and place in the project root.
---
## ⚙️ Methodology
1. Loaded and explored 284,807 real European cardholder transactions
2. Scaled `Amount` and `Time` using StandardScaler
3. Split into 80% train / 20% test with stratification
4. Applied **SMOTE** to fix the 0.172% fraud imbalance
5. Trained a **Random Forest Classifier** on the balanced data
6. Evaluated with **AUPRC** — the correct metric for imbalanced problems
7. Built a scoring system that returns a fraud probability and risk label per transaction
---
## 🚨 Anomaly Scoring System
| Fraud Score | Risk Level |
|---|---|
| 0.0 – 0.3 | 🟢 LOW |
| 0.3 – 0.7 | 🟡 MEDIUM |
| 0.7 – 1.0 | 🔴 HIGH |
---
## 📊 Evaluation
| Metric | Purpose |
|---|---|
| **AUPRC** | Primary metric — built for imbalanced data |
| **ROC-AUC** | Overall class separation |
| **Confusion Matrix** | Exact fraud caught vs missed |
> ❌ Accuracy is not used. A model predicting zero fraud scores 99.8% accuracy — completely useless in practice.
---
## 🗂 Project Structure

| File | Description |
|---|---|
| `1_explore.py` | Load and explore the dataset |
| `2_preprocess.py` | Scale features and split data |
| `3_train.py` | SMOTE + Random Forest training |
| `4_evaluate.py` | Performance metrics + PR curve |
| `5_score.py` | Anomaly scoring system |
| `outputs/pr_curve.png` | Precision-Recall chart |
| `README.md` | Project documentation |

---
## 🛠 Tools Used
- Python 3.x
- `pandas`, `numpy`
- `scikit-learn` — RandomForest, StandardScaler, metrics
- `imbalanced-learn` — SMOTE
- `matplotlib`, `seaborn`
- `joblib`
---
## ▶️ How to Run
Install dependencies:
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

Run scripts in order:

python 1_explore.py
python 2_preprocess.py
python 3_train.py
python 4_evaluate.py
python 5_score.py

💡 Why This Project Matters
Fraud detection is one of the highest-demand ML applications in finance. This project shows:

Handling severely imbalanced real-world data
Building a production-style anomaly scoring system
Choosing the right evaluation metrics
Writing clean, modular, readable code
👤 Author
Rehan Ali — Data Analyst

GitHub
