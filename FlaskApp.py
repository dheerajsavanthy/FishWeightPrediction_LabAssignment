# -*- coding: utf-8 -*-


import numpy as np
import pickle
import pandas as pd
from flask import Flask, request
from flask import Flask, request, jsonify, render_template
import os
cwd = os.getcwd()

app=Flask(__name__)
pickle_in = open("regressor.pkl","rb")
regressor=pickle.load(pickle_in)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = regressor.predict(final_features)

    
    return render_template('index.html', prediction_text='The weight of the fish is {}'.format(prediction))
    
    

if __name__=='__main__':
    app.run()