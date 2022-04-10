#Flask code here
from re import S
from typing import final
import numpy as np
from flask import Flask, app, request, render_template
import pickle
import joblib
from numpy.core.defchararray import array
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)




scaler_flask = MinMaxScaler()
logistic_model = joblib.load('logistic_reg.pkl')
svm_model = joblib.load('svc_rbf.pkl')
random_model = joblib.load('random_forest.pkl')
decision_model = joblib.load('descision_tree.pkl')
boosting_model = joblib.load('boosting.pkl')
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predictions', methods=["POST", "GET"])
def predictions():

    if request.method == 'POST':
        features = [int(x) for x in request.form.values()]

    final_features = np.asarray([features])
    print(final_features)
    predict = boosting_model.predict(final_features)
    print(predict)
    if predict==1:
        s = "THE PATIENT IS DIABETIC AND NEEDS TO CONSULT A DOCTOR"
        
    else:
        s = "THE PATIENT IS HEALTHY AND NON DIABETIC"
        

  
   
    
    return render_template('result.html', data=s)

@app.route('/about')
def about():
    return render_template('about.html')






if __name__ == "__main__":
    app.run(debug=True)