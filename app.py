from flask import Flask, request, render_template_string
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model pipeline (should include preprocessing)
model = joblib.load('model.pkl')

# Dropdown options
bank_options = ['Access', 'GTBank', 'UBA', 'FirstBank', 'Zenith']
card_types = ['Visa', 'MasterCard', 'Verve']
transaction_types = ['POS', 'Online', 'ATM']
locations = ['Abuja', 'Lagos', 'Kano', 'Port Harcourt']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Collect inputs
            data = pd.DataFrame([{
                'Transaction_Amount': float(request.form['amount']),
                'Bank_Name': request.form['bank'],
                'Card_Type': request.form['card'],
                'Transaction_Type': request.form['transaction'],
                'Location': request.form['location']
            }])

            # Predict
            result = model.predict(data)[0]
            prediction = "🔴 Fraudulent Transaction" if result == 1 else "🟢 Legitimate Transaction"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    # Render form
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Credit Card Fraud Detection</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(to right, #e0f7fa, #e1f5fe);
                font-family: 'Segoe UI', sans-serif;
                padding-top: 50px;
            }
            .container {
                max-width: 600px;
                background-color: white;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 8px 16px rgba(0,0,0,0.2);
            }
            .result {
                font-size: 1.2em;
                font-weight: bold;
                padding-top: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2 class="text-center mb-4">Credit Card Fraud Detection</h2>
            <form method="post">
                <div class="mb-3">
                    <label>Transaction Amount</label>
                    <input type="number" step="0.01" name="amount" class="form-control" required>
                </div>
                <div class="mb-3">
                    <label>Bank Name</label>
                    <select name="bank" class="form-select" required>
                        {% for bank in banks %}
                            <option value="{{ bank }}">{{ bank }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="mb-3">
                    <label>Card Type</label>
                    <select name="card" class="form-select" required>
                        {% for card in cards %}
                            <option value="{{ card }}">{{ card }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="mb-3">
                    <label>Transaction Type</label>
                    <select name="transaction" class="form-select" required>
                        {% for t in types %}
                            <option value="{{ t }}">{{ t }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div class="mb-3">
                    <label>Location</label>
                    <select name="location" class="form-select" required>
                        {% for l in locations %}
                            <option value="{{ l }}">{{ l }}</option>
                        {% endfor %}
                    </select>
                </div>
                <button type="submit" class="btn btn-primary w-100">Detect Fraud</button>
            </form>
            {% if prediction %}
                <div class="result text-center mt-4">{{ prediction }}</div>
            {% endif %}
        </div>
    </body>
    </html>
    ''', banks=bank_options, cards=card_types, types=transaction_types, locations=locations, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
