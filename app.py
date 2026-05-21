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

app = Flask(__name__)  # FIX #1: removed quotes around __name__

@app.route("/")
def load_page():
    return render_template('home.html', query="")
    
@app.route("/", methods=['POST'])
def cancerPrediction():
    dataset_url = "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/master/breast-cancer-data.csv"
    df = pd.read_csv(dataset_url)
    
    # FIX #2: Convert inputs to float
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
    proba = model.predict_proba(new_df)[:, 1]
    
    # FIX #3: Check single[0] instead of single
    if single[0] == 1:
        output1 = "The patient is diagnosed with Breast Cancer"
        output2 = "Confidence: {:.2f}%".format(proba[0] * 100)
    else:
        output1 = "The patient is not diagnosed with Breast Cancer"
        output2 = ""
    
    return render_template('html.html', 
                           output2=output2, 
                           query1=request.form['query1'], 
                           query2=request.form['query2'], 
                           query3=request.form['query3'], 
                           query4=request.form['query4'], 
                           query5=request.form['query5'])
if __name__ == "__main__":
    app.run( host="0.0.0.0", debug=True, use_reloader=False)