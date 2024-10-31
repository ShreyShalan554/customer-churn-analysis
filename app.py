# coding: utf-8

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn import metrics
from flask import Flask, request, render_template
import pickle
import os

app = Flask("__name__")

# Load your dataset once when the app starts
df_1 = pd.read_csv("first_telc.csv")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")

@app.route("/", methods=['POST'])
def predict():
    '''
    SeniorCitizen
    MonthlyCharges
    TotalCharges
    gender
    Partner
    Dependents
    PhoneService
    MultipleLines
    InternetService
    OnlineSecurity
    OnlineBackup
    DeviceProtection
    TechSupport
    StreamingTV
    StreamingMovies
    Contract
    PaperlessBilling
    PaymentMethod
    tenure
    '''
    
    # Gather input data from the form
    input_data = [request.form[f'query{i}'] for i in range(1, 20)]

    # Load the model
    model = pickle.load(open("model.sav", "rb"))
    
    # Prepare the input data for prediction
    new_df = pd.DataFrame([input_data], columns=[
        'SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 
        'PaymentMethod', 'tenure'
    ])
    
    df_2 = pd.concat([df_1, new_df], ignore_index=True)
    
    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    df_2['tenure_group'] = pd.cut(df_2.tenure.astype(int), range(1, 80, 12), right=False, labels=labels)
    df_2.drop(columns=['tenure'], axis=1, inplace=True)
    
    new_df_dummies = pd.get_dummies(df_2[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                                             'PhoneService', 'MultipleLines', 'InternetService', 
                                             'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                             'TechSupport', 'StreamingTV', 'StreamingMovies', 
                                             'Contract', 'PaperlessBilling', 'PaymentMethod', 
                                             'tenure_group']])
    
    # Make prediction
    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]
    
    if single == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {:.2f}%".format(probability * 100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {:.2f}%".format(probability * 100)
        
    return render_template('home.html', output1=o1, output2=o2, **{f'query{i}': request.form[f'query{i}'] for i in range(1, 20)})

if __name__ == "__main__":
    # Get the port from the environment variable
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host='0.0.0.0', port=port)  # Bind to all interfaces
