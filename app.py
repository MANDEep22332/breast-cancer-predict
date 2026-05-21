# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:23:24 2026

@author: mandeep
"""

import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, render_template

app = Flask(__name__) 

@app.route("/")
def load_page():
    return render_template('home.html', query="")
    
@app.route("/", methods=['POST'])
def cancerPrediction():
    dataset_url = "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/master/breast-cancer-data.csv"
    df = pd.read_csv(dataset_url)
    
    # Convert inputs to float
    inputQuery1 = float(request.form['query1'])
    inputQuery2 = float(request.form['query2'])
    inputQuery3 = float(request.form['query3'])
    inputQuery4 = float(request.form['query4'])
    inputQuery5 = float(request.form['query5'])
    
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    features = ['radius_mean', 'texture_mean', 'perimeter_mean', 'smoothness_mean', 'compactness_mean']
    X = df[features]
    y = df['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    model.fit(X_train, y_train)
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
    new_df = pd.DataFrame(data, columns=features)
    
    single = model.predict(new_df)
    proba = model.predict_proba(new_df)
    
    if single[0] == 1:
        output1 = "The patient is diagnosed with Breast Cancer"
        # proba[0][1] is the confidence for Malignant (1)
        output2 = "Confidence: {:.2f}%".format(proba[0][1] * 100)
    else:
        output1 = "The patient is not diagnosed with Breast Cancer"
        # proba[0][0] is the confidence for Benign (0)
        output2 = "Confidence: {:.2f}%".format(proba[0][0] * 100)
    
    # FIXED: Added output1=output1 down below
    return render_template('html.html', 
                           output1=output1, 
                           output2=output2, 
                           query1=request.form['query1'], 
                           query2=request.form['query2'], 
                           query3=request.form['query3'], 
                           query4=request.form['query4'], 
                           query5=request.form['query5'])

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, use_reloader=False)