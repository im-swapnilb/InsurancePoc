#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
import flask as Flask
from flask import request
from flask import render_template
import pickle


# In[12]:


from flask import Flask

app = Flask(__name__)


# In[13]:


@app.route('/')
def home():
    return render_template('InsuranceModel.html')


# In[14]:


# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,5)
    loaded_model = pickle.load(open("insurance_exp_flask.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    print(result[0])
    return result[0]


# In[16]:


@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        if to_predict_list[1] == 'Male':
            to_predict_list[1] = 0
        else:
            to_predict_list[1] = 1
        if to_predict_list[4] == 'yes':
            to_predict_list[4] = 1
        else:
            to_predict_list[4] = 0

        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)
        return render_template("result.html", prediction = result)


# In[17]:


# Main function
if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOADED'] = True

