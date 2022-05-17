# -*- coding: utf-8 -*-
"""
Created on Tue May 17 14:52:17 2022

Load and Deploy model

@author: Marvin
"""

import re
import pandas as pd
import numpy as np
import datetime
import os
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model

#%%

PKL_NAME = os.path.join(os.getcwd(), 'best_model.pkl')

# load saved model
with open(PKL_NAME, 'rb') as file:
    model = pickle.load(file)
    
# Load x_test, and y_test data before testing model
#score = best_model.score(X_test, y_test)
#print(score*100, '%')

#If using Machine Learning load model
#model = pickle.load(open(PKL_NAME))

heart_attack_chance = {0: 'Negative', 1: 'Positive'}


#%% Deployment

# test static data:
#patient_info = np.array([45,0,1,128,204,0,0,172,0,1.4,2,0,2])
#patient_info_ex = np.expand_dims(patient_info, axis=0)                                                             

#outcome = model.predict(patient_info_ex)
#print(outcome)

#print(heart_attack_chance[np.argmax(outcome)])


#%% build app using streamlit to take in input from the webpage

#age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak',
#'slp','caa','thall','output


with st.form('Heart Attack Analysus Prediction Form'):
    st.write("Patient's information")
    # all features in the form
    age = int(st.number_input('What is your age?'))
    sex = int(st.number_input('Gender 0 (Male), 1(Female)'))
    cp = int(st.number_input("""Chest Pain type\n 
                  (0: typical angina), \n  
                  (1: atypical angina), \n
                  (2: non-anginal pain), \n
                  (3: asymptomatic) """))
    trtbps = int(st.number_input('Resting blood pressure (in mm Hg)'))
    chol = int(st.number_input('Cholestoral in mg/dl'))
    fbs = int(st.number_input('fasting blood sugar > 120 mg/dl(1 = true; 0 = false)'))
    restecg = int(st.number_input("""Resting electrocardiographic results\n 
                  (0: normal), \n  
                  (1: having ST-T wave abnormality), \n
                  (2: showing probable or definite left ventricular hypertrophy"""))
    thalachh = int(st.number_input('Maximum heart rate achieved'))
    exng = int(st.number_input('exercise induced angina (1 = yes; 0 = no)'))
    oldpeak = int(st.number_input('oldpeak'))
    slp = int(st.number_input('Slp? (0 = yes; 1 = no, 2 = maybe)'))
    caa = int(st.number_input('number of major vessels (0-3)'))
    thall = int(st.number_input('thall (1-3)'))
    

    
    submitted = st.form_submit_button('Submit')
    
    if submitted == True:
        # take user input from the form as 13 features
        patient_info = np.array([age,sex,cp,trtbps,chol,fbs,restecg,
                                 thalachh,exng,oldpeak,slp,caa,thall])
        patient_info_ex = np.expand_dims(patient_info, axis=0)
       
        outcome = model.predict(patient_info_ex)
        #print out the outcome in the form using write function
        st.write(heart_attack_chance[np.argmax(outcome)])
            
        
        if np.argmax(outcome)==1:
            st.warning("""You have higher change to have heart attack, \n
                       please exercise more and eat healthy diet""")
        else:
            st.balloons()
            st.success('you have a healthy heart! keep up the exercise and healthy diet')
    
    st.write(submitted)

# to deploy need to run the script below in the tf_env where the .py file is stored
# streamlit run Deploy_model_diabetes.py
# (tf_env) C:\<path of the saved deploy_heart_pred_app.py file>streamlit run Deploy_heart_pred_app.py


#%%
