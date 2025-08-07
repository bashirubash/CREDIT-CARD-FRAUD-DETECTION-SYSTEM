# app.py

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open('fraud_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        inputs = [
            float(request.form['amount']),
            int(request.form['transaction_hour']),
            int(request.form['device_type']),
            int(request.form['channel']),
            int(request.form['merchant_type']),
            int(request.form['customer_age']),
            int(request.form['location_code']),
        ]
        prediction = model.predict([inputs])[0]
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        return f"Error: {e}"
@app.route('/dashboard')
def dashboard():
    # Sample fraud data
    fraud_logs = [
        {"date": "2025-08-01", "location": "Lagos", "amount": 30500, "status": "Fraud"},
        {"date": "2025-08-02", "location": "Abuja", "amount": 15000, "status": "Normal"},
        {"date": "2025-08-03", "location": "Kano", "amount": 72000, "status": "Fraud"},
        {"date": "2025-08-03", "location": "Port Harcourt", "amount": 8200, "status": "Normal"},
    ]

    # Chart data
    chart_labels = ["Lagos", "Abuja", "Kano", "Port Harcourt"]
    chart_data = [5, 1, 3, 2]  # Number of frauds per location (mocked)

    return render_template('dashboard.html', fraud_logs=fraud_logs, chart_labels=chart_labels, chart_data=chart_data)

if __name__ == '__main__':
    app.run(debug=True)

