from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

# Define label encoders or mappings
transaction_map = {'POS': 0, 'Transfer': 1, 'ATM': 2, 'USSD': 3, 'Internet': 4}
bank_map = {'Access': 0, 'Opay': 1, 'Kuda': 2, 'MoneyPoint': 3, 'Zenith': 4, 'GTB': 5, 'UBA': 6, 'First Bank': 7}
time_map = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
location_map = {'Lagos': 0, 'Yola': 1, 'Abuja': 2, 'Port Harcourt': 3, 'Kano': 4}  # You can add more

@app.route('/')
def form():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    amount = float(request.form['amount'])
    transaction_type = transaction_map.get(request.form['transaction_type'], 0)
    bank_name = bank_map.get(request.form['bank_name'], 0)
    time_of_day = time_map.get(request.form['time_of_day'], 0)
    location_input = request.form['location']
    location = location_map.get(location_input, 0)  # default to 0 if unknown

    # Create feature vector
    features = np.array([[amount, transaction_type, bank_name, time_of_day, location]])
    
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    result = "FRAUDULENT TRANSACTION 🚨" if prediction == 1 else "GENUINE TRANSACTION ✅"
    
    return render_template(
        'result.html',
        result=result,
        probability=round(proba * 100, 2),
        amount=amount,
        location=location_input
    )

if __name__ == '__main__':
    app.run(debug=True)
