# Importing the libraries required

from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model

filename = 'heart_failure_prediction_model.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        cp = int(request.form['creatinine_phosphokinase'])
        ef = int(request.form['ejection_fraction'])
        hbp = int(request.form['high_blood_pressure'])
        plt = int(request.form['platelets'])
        sc = float(request.form['serum_creatinine'])
        time = int(request.form['time'])
        
        data = np.array([[age, cp, ef, hbp, plt, sc, time]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)