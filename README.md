# 🔍 Credit Card Fraud Detection
📌 **Business Problem**
Banks and fintech companies lose billions to fraudulent transactions every year. This project builds a machine learning pipeline that detects fraud in real time and assigns every transaction an anomaly risk score — a critical tool for modern banking and fintech.
---
## 📁 Dataset
**Source:** [Kaggle — Credit Card Fraud Detection by MLG-ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
| File | Description |
|---|---|
| creditcard.csv | 284,807 real transactions with 30 features and a fraud label |
> ⚠️ Dataset not included in this repo due to file size. Download it from Kaggle and place it in the project root folder.
---
## ⚙️ Methodology
- Loaded and explored 284,807 real European credit card transactions
- Scaled `Amount` and `Time` features using StandardScaler
- Split data into 80% train / 20% test with stratification
- Applied **SMOTE** to fix the extreme class imbalance (only 0.172% fraud)
- Trained a **Random Forest Classifier** on the balanced dataset
- Evaluated using **AUPRC** — the correct metric for imbalanced classification
- Built an **anomaly scoring system** that assigns each transaction a fraud probability and risk level
---
## 🚨 Anomaly Scoring System
Every transaction receives a fraud probability score from 0 to 1:
| Fraud Score | Risk Level |
|---|---|
| 0.0 – 0.3 | 🟢 LOW |
| 0.3 – 0.7 | 🟡 MEDIUM |
| 0.7 – 1.0 | 🔴 HIGH |
---
## 📊 Evaluation Metrics
| Metric | Why It Matters |
|---|---|
| AUPRC | Primary metric — handles class imbalance correctly |
| ROC-AUC | Measures overall ability to separate fraud from legit |
| Confusion Matrix | Shows exact count of fraud caught vs missed |
> ❌ Accuracy is NOT used. A model that flags zero fraud still scores 99.8% accuracy due to the imbalance — making it completely useless in practice.
---
## 🗂 Project Structure

fraud-detection/
├── 1_explore.py # Load and understand the dataset
├── 2_preprocess.py # Scale features and split into train/test
├── 3_train.py # Handle imbalance with SMOTE + train model
├── 4_evaluate.py # Evaluate performance and generate PR curve
├── 5_score.py # Anomaly scoring system with risk levels
├── outputs/
│ └── pr_curve.png # Precision-Recall curve chart
└── README.md

---
## 🛠 Tools Used
- Python 3.x
- pandas, numpy
- scikit-learn (RandomForestClassifier, StandardScaler, metrics)
- imbalanced-learn (SMOTE)
- matplotlib, seaborn
- joblib
---
## ▶️ How to Run

pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn joblib

Run scripts in order:

python 1_explore.py
python 2_preprocess.py
python 3_train.py
python 4_evaluate.py
python 5_score.py

💡 Why This Project Matters
Fraud detection is one of the most in-demand applications of machine learning in the finance industry. This project demonstrates:

Handling severely imbalanced datasets (a real-world challenge)
Building a production-style scoring system (not just a model)
Using the right evaluation metrics (AUPRC over accuracy)
Clean, modular code split across focused scripts
👤 Author
Rehan Ali — Data Analyst

GitHub
