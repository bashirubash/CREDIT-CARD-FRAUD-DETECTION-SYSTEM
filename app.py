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

if __name__ == '__main__':
    app.run(debug=True)
