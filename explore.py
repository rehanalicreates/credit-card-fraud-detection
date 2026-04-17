import pandas as pd

df = pd.read_csv('creditcard.csv')

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nClass distribution:")
print(df['Class'].value_counts())
print(f"\nFraud percentage: {df['Class'].mean()*100:.3f}%")
print("\nMissing values:", df.isnull().sum().sum())