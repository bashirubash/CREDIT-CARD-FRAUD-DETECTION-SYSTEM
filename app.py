from flask import Flask, request, render_template_string
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load dataset and preprocess
df = pd.read_csv('nigerian_credit_card_fraud_dataset.csv')

# Select only the required 5 features
selected_features = ['Amount', 'Transaction Type', 'Bank Name', 'Time of Day', 'Transaction Location']
df = df[selected_features + ['Is Fraud?']]

# Encode categorical features
label_encoders = {}
for col in ['Transaction Type', 'Bank Name', 'Time of Day', 'Transaction Location']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Train model
X = df[selected_features]
y = df['Is Fraud?']
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model and encoders for reuse
joblib.dump(model, 'model.pkl')
joblib.dump(label_encoders, 'encoders.pkl')

# HTML Form Template
form_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Credit Card Fraud Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="bg-light">
<div class="container mt-5">
    <div class="card shadow rounded">
        <div class="card-header bg-primary text-white text-center">
            <h2>Detect Credit Card Fraud</h2>
        </div>
        <div class="card-body">
            <form method="post" action="/predict">
                <div class="mb-3">
                    <label for="amount" class="form-label">Amount</label>
                    <input type="number" step="0.01" name="amount" class="form-control" required>
                </div>
                <div class="mb-3">
                    <label for="transaction_type" class="form-label">Transaction Type</label>
                    <select name="transaction_type" class="form-select" required>
                        <option value="POS">POS</option>
                        <option value="Transfer">Transfer</option>
                        <option value="ATM">ATM</option>
                        <option value="USSD">USSD</option>
                        <option value="Internet">Internet</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="bank_name" class="form-label">Bank Name</label>
                    <select name="bank_name" class="form-select" required>
                        <option value="Access">Access</option>
                        <option value="Opay">Opay</option>
                        <option value="Kuda">Kuda</option>
                        <option value="MoneyPoint">MoneyPoint</option>
                        <option value="Zenith">Zenith</option>
                        <option value="GTB">GTB</option>
                        <option value="UBA">UBA</option>
                        <option value="First Bank">First Bank</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="time_of_day" class="form-label">Time of Day</label>
                    <select name="time_of_day" class="form-select" required>
                        <option value="Morning">Morning</option>
                        <option value="Afternoon">Afternoon</option>
                        <option value="Evening">Evening</option>
                        <option value="Night">Night</option>
                    </select>
                </div>
                <div class="mb-3">
                    <label for="transaction_location" class="form-label">Transaction Location</label>
                    <input type="text" name="transaction_location" class="form-control" placeholder="e.g., Yola" required>
                </div>
                <button type="submit" class="btn btn-success w-100">Detect Fraud</button>
            </form>
        </div>
    </div>
</div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(form_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model and encoders
        model = joblib.load('model.pkl')
        encoders = joblib.load('encoders.pkl')

        # Collect form data
        amount = float(request.form['amount'])
        transaction_type = request.form['transaction_type']
        bank_name = request.form['bank_name']
        time_of_day = request.form['time_of_day']
        location = request.form['transaction_location']

        # Encode categorical values
        encoded_data = [
            amount,
            encoders['Transaction Type'].transform([transaction_type])[0],
            encoders['Bank Name'].transform([bank_name])[0],
            encoders['Time of Day'].transform([time_of_day])[0],
            encoders['Transaction Location'].transform([location])[0] if location in encoders['Transaction Location'].classes_ else 0
        ]

        # Predict
        prediction = model.predict([encoded_data])[0]

        result = "❌ Fraudulent Transaction Detected!" if prediction == 1 else "✅ Transaction is Safe."
        return f"<h2 style='text-align:center;margin-top:50px;'>{result}</h2><div style='text-align:center;'><a href='/'>&larr; Try another</a></div>"

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3><a href='/'>Try again</a>"

if __name__ == '__main__':
    app.run(debug=True)
