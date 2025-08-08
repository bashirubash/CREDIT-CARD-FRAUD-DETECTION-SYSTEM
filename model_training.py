import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("nigerian_credit_card_fraud_dataset.csv")

# Select relevant features
features = ['Transaction_Amount', 'Bank_Name', 'Card_Type', 'Transaction_Type', 'Location']
target = 'Is_Fraud'  # Assuming this is the fraud label column name

X = df[features]
y = df[target]

# Define preprocessing
numeric_features = ['Transaction_Amount']
numeric_transformer = StandardScaler()

categorical_features = ['Bank_Name', 'Card_Type', 'Transaction_Type', 'Location']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
clf.fit(X_train, y_train)

# Save the model pipeline
joblib.dump(clf, 'model.pkl')
print("✅ Model trained and saved as model.pkl")
