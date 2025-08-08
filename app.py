from flask import Flask, request, render_template_string
import pickle
import numpy as np

# Load model and label encoders
with open('fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

app = Flask(__name__)

# HTML + CSS + Bootstrap form (inline)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Credit Card Fraud Detection</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: linear-gradient(to right, #e0f7fa, #fff); min-height: 100vh; display: flex; align-items: center; justify-content: center; }
        .container { background-color: #ffffff; padding: 40px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); }
        h2 { text-align: center; margin-bottom: 30px; }
    </style>
</head>
<body>
    <div class="container">
        <h2>Credit Card Fraud Detection</h2>
        <form method="post">
            <div class="mb-3">
                <label>Amount</label>
                <input type="number" step="any" name="amount" class="form-control" required>
            </div>
            <div class="mb-3">
                <label>Location</label>
                <select name="location" class="form-control" required>
                    <option value="">Select</option>
                    <option value="Lagos">Lagos</option>
                    <option value="Abuja">Abuja</option>
                    <option value="Kano">Kano</option>
                </select>
            </div>
            <div class="mb-3">
                <label>Channel</label>
                <select name="channel" class="form-control" required>
                    <option value="">Select</option>
                    <option value="POS">POS</option>
                    <option value="ATM">ATM</option>
                    <option value="Online">Online</option>
                </select>
            </div>
            <div class="mb-3">
                <label>Time</label>
                <select name="time" class="form-control" required>
                    <option value="">Select</option>
                    <option value="Morning">Morning</option>
                    <option value="Afternoon">Afternoon</option>
                    <option value="Evening">Evening</option>
                    <option value="Night">Night</option>
                </select>
            </div>
            <div class="mb-3">
                <label>Merchant</label>
                <select name="merchant" class="form-control" required>
                    <option value="">Select</option>
                    <option value="Shoprite">Shoprite</option>
                    <option value="Jumia">Jumia</option>
                    <option value="Konga">Konga</option>
                </select>
            </div>
            <button type="submit" class="btn btn-primary w-100">Detect Fraud</button>
        </form>
        {% if result is not none %}
            <div class="alert alert-{{ 'danger' if result == 1 else 'success' }} mt-4 text-center">
                <strong>{{ 'Fraudulent Transaction Detected!' if result == 1 else 'Transaction Seems Legitimate.' }}</strong>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            # Get form data
            amount = float(request.form['amount'])
            location = request.form['location']
            channel = request.form['channel']
            time = request.form['time']
            merchant = request.form['merchant']

            # Prepare input for model
            input_data = [amount]
            for feature, value in zip(['location', 'channel', 'time', 'merchant'],
                                      [location, channel, time, merchant]):
                encoder = label_encoders[feature]
                encoded_value = encoder.transform([value])[0]
                input_data.append(encoded_value)

            # Predict
            prediction = model.predict([input_data])[0]
            result = int(prediction)

        except Exception as e:
            result = None
            print(f"❌ Error: {e}")

    return render_template_string(HTML_TEMPLATE, result=result)

if __name__ == '__main__':
    app.run(debug=True)
