from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('insurance_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        age = float(request.form['age'])
        sex = float(request.form['sex'])  # 0 for male, 1 for female
        bmi = float(request.form['bmi'])
        children = float(request.form['children'])
        smoker = float(request.form['smoker'])  # 0 for yes, 1 for no
        region = float(request.form['region'])  # 0:southeast, 1:southwest, 2:northeast, 3:northwest

        # Make prediction
        input_data = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(input_data)[0]

        return render_template('output.html', prediction=prediction)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)