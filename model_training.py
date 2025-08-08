import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv('nigerian_credit_card_fraud_dataset.csv')

# Keep only necessary columns
features = ['amount', 'bank', 'transaction_type', 'time', 'location']
target = 'is_fraud'  # or whatever your label column is

df = df[features + [target]]

# Encode categorical columns
label_encoders = {}
for col in ['bank', 'transaction_type', 'location']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Save the label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')

# Split features and target
X = df[features]
y = df[target]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate (optional)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(model, 'model.pkl')
