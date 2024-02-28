import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load the models and scaler
standard_scaler = pickle.load(open('standardscaler.pkl', 'rb'))
decision_tree_1 = pickle.load(open('decisiontree1.pkl', 'rb'))
decision_tree_2 = pickle.load(open('decisiontree2.pkl', 'rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Extracting data from the form
        X1 = float(request.form.get('Relative Compactness'))
        X2 = float(request.form.get('Surface Area'))
        X3 = float(request.form.get('Wall Area'))
        X4 = float(request.form.get('Roof Area'))
        X5 = float(request.form.get('Overall Height'))
        X6 = float(request.form.get('Orientation'))
        X7 = float(request.form.get('Glazing Area'))
        X8 = float(request.form.get('Glazing Area Distribution'))

        new_data = [[X1, X2, X3, X4, X5, X6, X7, X8]]
        
        # Scaling the data
        new_data_scaled = standard_scaler.transform([new_data])
        
        # Predicting with the models
        result1 = decision_tree_1.predict(new_data_scaled)
        result2 = decision_tree_2.predict(new_data_scaled)

        return render_template('form.html', result1=result1[0], result2=result2[0])

    else:
        # Handle GET request (if needed)
        return render_template('form.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")

