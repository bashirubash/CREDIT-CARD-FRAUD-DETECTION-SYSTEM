# model_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Nigerian dataset
df = pd.read_csv('nigerian_credit_card_fraud_dataset.csv')

# Drop irrelevant columns
X = df.drop(columns=['is_fraud', 'location', 'merchant'])
y = df['is_fraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('fraud_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("✅ Model trained and saved as fraud_model.pkl")
