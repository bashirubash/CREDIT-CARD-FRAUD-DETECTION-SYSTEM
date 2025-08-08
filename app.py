from flask import Flask, request, render_template_string
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

app = Flask(__name__)

# Load dataset
df = pd.read_csv("nigerian_credit_card_fraud_dataset.csv")

# Drop missing values
df.dropna(inplace=True)

# Encode categorical columns
le_bank = LabelEncoder()
le_type = LabelEncoder()
le_location = LabelEncoder()
le_time = LabelEncoder()

df['Bank'] = le_bank.fit_transform(df['Bank'])
df['TransactionType'] = le_type.fit_transform(df['TransactionType'])
df['Location'] = le_location.fit_transform(df['Location'])
df['TimeOfDay'] = le_time.fit_transform(df['TimeOfDay'])

# Features and target
X = df[['Bank', 'TransactionType', 'Location', 'Amount', 'TimeOfDay']]
y = df['IsFraud']  # 1 = Fraud, 0 = Legit

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# HTML Template with Bootstrap
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Credit Card Fraud Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(to right, #f0f4f7, #cfe0f5);
            padding: 50px;
        }
        .form-container {
            background-color: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px #ccc;
            max-width: 600px;
            margin: auto;
        }
    </style>
</head>
<body>
    <div class="form-container">
        <h2 class="text-center mb-4">Credit Card Fraud Detection</h2>
        <form method="post">
            <div class="mb-3">
                <label for="bank" class="form-label">Bank Name</label>
                <select class="form-select" name="bank" required>
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
                <label for="transaction_type" class="form-label">Transaction Type</label>
                <select class="form-select" name="transaction_type" required>
                    <option value="POS">POS</option>
                    <option value="Transfer">Transfer</option>
                    <option value="ATM">ATM</option>
                    <option value="USSD">USSD</option>
                    <option value="Internet">Internet</option>
                </select>
            </div>
            <div class="mb-3">
                <label for="location" class="form-label">Transaction Location</label>
                <input type="text" class="form-control" name="location" placeholder="e.g., Abuja" required>
            </div>
            <div class="mb-3">
                <label for="amount" class="form-label">Amount (₦)</label>
                <input type="number" class="form-control" name="amount" required>
            </div>
            <div class="mb-3">
                <label for="time" class="form-label">Time of Day</label>
                <select class="form-select" name="time" required>
                    <option value="Morning">Morning</option>
                    <option value="Afternoon">Afternoon</option>
                    <option value="Evening">Evening</option>
                    <option value="Night">Night</option>
                </select>
            </div>
            <button type="submit" class="btn btn-primary w-100">Check Fraud Status</button>
        </form>
        {% if prediction %}
        <div class="alert alert-info mt-4 text-center">
            <strong>Prediction Result:</strong> {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    if request.method == "POST":
        bank = request.form["bank"]
        transaction_type = request.form["transaction_type"]
        location = request.form["location"]
        amount = float(request.form["amount"])
        time_of_day = request.form["time"]

        # Encode inputs
        bank_encoded = le_bank.transform([bank])[0] if bank in le_bank.classes_ else 0
        type_encoded = le_type.transform([transaction_type])[0] if transaction_type in le_type.classes_ else 0
        location_encoded = le_location.transform([location])[0] if location in le_location.classes_ else 0
        time_encoded = le_time.transform([time_of_day])[0] if time_of_day in le_time.classes_ else 0

        features = [[bank_encoded, type_encoded, location_encoded, amount, time_encoded]]
        result = model.predict(features)[0]
        prediction = "⚠️ Fraudulent Transaction" if result == 1 else "✅ Legitimate Transaction"

    return render_template_string(html, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
