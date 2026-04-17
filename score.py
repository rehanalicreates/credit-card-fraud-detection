import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('creditcard.csv')
scaler = joblib.load('scaler.pkl')
df['Amount_scaled'] = scaler.transform(df[['Amount']])
df['Time_scaled'] = scaler.transform(df[['Time']])
df = df.drop(['Time', 'Amount'], axis=1)

X = df.drop('Class', axis=1)
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = joblib.load('fraud_model.pkl')

def anomaly_score(features):
    prob = model.predict_proba([features])[0][1]
    if prob < 0.3:
        risk = "LOW"
    elif prob < 0.7:
        risk = "MEDIUM"
    else:
        risk = "HIGH"
    return {
        "fraud_score": round(prob, 4),
        "risk_level": risk,
        "is_fraud": prob >= 0.5
    }

# Test on 5 transactions from test set
print("Scoring 5 sample transactions:\n")
for i in range(5):
    features = X_test.iloc[i].values
    actual = y_test.iloc[i]
    result = anomaly_score(features)
    print(f"Transaction {i+1}: Score={result['fraud_score']} | Risk={result['risk_level']} | Actual={'FRAUD' if actual==1 else 'LEGIT'}")