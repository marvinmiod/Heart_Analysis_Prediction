# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:23:07 2022

Heart attack predictor model

This model predict the probability of one’s getting heart attack 
can be determined by analysing the patient’s age, gender, exercise 
induced angina, number of major vessels, 
chest pain indication, resting blood pressure, cholesterol level, 
fasting blood sugar, resting electrocardiographic results, 
and maximum heart rate achieved.

@author: Marvin Miod
"""

import re
import pandas as pd
import numpy as np
import re
import datetime
import os
import pickle
import missingno as msno
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
import seaborn as sns



#%%

# save the model and save the log
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'heart_attack_predictor.h5')
# path where the logfile for tensorboard call back
LOG_PATH = os.path.join(os.getcwd(),'Log_heart_attack')
log_files = os.path.join(LOG_PATH, 
                         datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
DATASET = os.path.join(os.getcwd(), 'dataset', 'heart.csv')

#PKL_NAME = os.path.join(os.getcwd(), 'best_model.pkl')
PKL_NAME = 'best_model.pkl'

#%%

# EDA
# Step 1) Load data
df = pd.read_csv(DATASET)
# create dummy data to work with
dummy_df = df.copy()

# Step 2) Data Inspection
df.head()
df.describe().T # check for outlier
df.dtypes
df.info()
# visualise the data
df.boxplot()
# To check/visualise if there is any missing data
df.isna().sum()

# to visualize the missing numbers in the dataframe
msno.matrix(df)
msno.bar(df)

column_names = ['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak',
'slp','caa','thall','output']

data = pd.DataFrame(dummy_df)
data.columns = column_names
cor = data.corr()
plt.figure(figsize=(12,10))
sns.heatmap(cor,annot=True, cmap=plt.cm.Reds)
plt.show() 

# Step 3) Clean Data
# Data cleaning not required because there is no NaN or missing values


# Step 4) Features selection # this is a binary classification, outcome 0 or 1
# based on the heatmap plot, the highest features that contribute 
# to heart attack is cp, thalachh, and slp with value of 0.43, 0.42. 0.35 respectively

# identify x features
#x1 = dummy_df['cp'] # 0.43 
#x2 = dummy_df['thalachh'] # 0.42
#x3 = dummy_df['slp'] # 0.35
#x = [x1,x2,x3]
#x_features = np.array(x).T 

# uncomment for all features
x_features = dummy_df.drop(columns='output') 

# target (y) is the output column
y_target = dummy_df['output']

#%% # Step 5) Data Pre-processing 
# using ML pipeline to determine which scaling to use


#%%# Machine Learning Pipeline

# prepare x and y train and test data
X_train,X_test, y_train,y_test = train_test_split(x_features, y_target, 
                                                  test_size = 0.3, 
                                                  random_state=12)

# machine learning pipeline using standard scaler and PCA
steps_tree = [('MinMax', StandardScaler()), 
    ('PCA', PCA(n_components=2)),
    ('Tree', DecisionTreeClassifier())]

steps_forest = [('MinMax', StandardScaler()), 
    ('PCA', PCA(n_components=2)),
    ('Forest', RandomForestClassifier())]

steps_logis = [('MinMax', StandardScaler()), 
    ('PCA', PCA(n_components=2)),
    ('Logic', LogisticRegression(solver='liblinear'))]

steps_svc = [('MinMax', StandardScaler()), 
    ('PCA', PCA(n_components=2)),
    ('SVC', SVC())]

steps_knn = [('MinMax', StandardScaler()), 
    ('PCA', PCA(n_components=2)),
    ('KNN', KNeighborsClassifier())]

tree_pipeline = Pipeline(steps_tree)
forest_pipeline = Pipeline(steps_forest)
logic_pipeline = Pipeline(steps_logis)
svc_pipeline = Pipeline(steps_svc)
knn_pipeline = Pipeline(steps_knn)

pipelines = [tree_pipeline, forest_pipeline, logic_pipeline, svc_pipeline,
             knn_pipeline]

for pipe in pipelines:
    pipe.fit(X_train,y_train)
    
best_score = 0
best_model = 0
best_pipeline = ''
    
pipe_dict = {0:'Tree', 1:'Forest',2:'Logistic Regression', 3:'SVC', 4:'KNN'}
print('Model Accuracy using Standard Scaler with PCA\n')


# Model analysis
for index, model in enumerate(pipelines):
    #print(model.score(X_test,y_test))
    print("{} Test Accuracy (%): {}".format(pipe_dict[index],model.score(X_test, 
                                                                     y_test)*100))
    if model.score(X_test,y_test) > best_score:
        best_score = model.score(X_test,y_test)        
        best_model = index
        best_pipeline = model 
        
        
print('\nBest score for Heart predictor model will be {} with % accuracy of {}' 
      .format(pipe_dict[best_model],best_score*100 ))     

# in this run the best model is Logistic Regression model with accuracy of 83%


#%% Summary report

#print the accuracy numbers in the console
pred_x = model.predict(X_test)
#y_true = np.argmax(y_test, axis=1)
#y_pred = np.argmax(pred_x, axis=1)
y_true = y_test
y_pred = pred_x
    
cm = confusion_matrix(y_true, y_pred)
cr = classification_report(y_true, y_pred)
print(cr)
    
# Code for Confusion Matrix correlation graph
#labels = [ '0', '1', '2' ,'3', '4', '5', '6' ,'7', '8', '9'] # manual way
labels = [str(i) for i in range(10)] # using list comprehension
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
#                              display_labels=np.unique(y_true))
    
# this one is to removed the display_labels with unique(y_true)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)

#plot the quadrant
disp.plot(cmap=plt.cm.Blues)
plt.show() 


#%% Save the best model (Logistic Regression model)

#PKL_NAME = 'best_model.pkl'

with open(PKL_NAME, 'wb') as file:
    pickle.dump(best_pipeline, file)  

