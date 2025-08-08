import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Load the trained model and encoders
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('channel_encoder.pkl', 'rb') as file:
    channel_encoder = pickle.load(file)

with open('location_encoder.pkl', 'rb') as file:
    location_encoder = pickle.load(file)

with open('merchant_encoder.pkl', 'rb') as file:
    merchant_encoder = pickle.load(file)

with open('time_encoder.pkl', 'rb') as file:
    time_encoder = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        amount = float(request.form['amount'])
        location = request.form['location']
        channel = request.form['channel']
        merchant = request.form['merchant']
        time = request.form['time']

        # Transform categorical inputs using encoders
        location_encoded = location_encoder.transform([location])[0]
        channel_encoded = channel_encoder.transform([channel])[0]
        merchant_encoded = merchant_encoder.transform([merchant])[0]
        time_encoded = time_encoder.transform([time])[0]

        # Combine inputs into a feature array
        features = np.array([[amount, location_encoded, channel_encoded, merchant_encoded, time_encoded]])

        # Make prediction
        prediction = model.predict(features)[0]
        result = "Fraudulent Transaction Detected ❌" if prediction == 1 else "Legitimate Transaction ✅"

        return render_template('result.html', result=result)

    except Exception as e:
        flash(f"Error: {e}", "danger")
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
