# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 18:06:41 2022

@author: SARKAR
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
st.write("energy prediction ccpp")

df = pd.read_csv('energy_production (2).csv',sep=";")

st.write("all the variables")
df
df.rename(columns={"temperature":"AT" , "exhaust_vacuum":"V" , "amb_pressure":"AP" , "r_humidity":"RH" , "energy_production": "PE"}, inplace=True)
df_4 = df[['AT', 'V', 'AP', 'RH']]
y = df['PE']

X_train, X_test, y_train, y_test = train_test_split(df_4, y, test_size = 0.2, random_state = 42)
rf_regressor=RandomForestRegressor(n_estimators=102,
    min_samples_split=20,
    min_samples_leaf=28,
    max_features='auto')
rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)
diff=y_test-y_pred
sns.distplot(diff)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score:', metrics.r2_score(y_test, y_pred))  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

import pickle
file=open("rf.pkl",'wb')
pickle.dump(rf_regressor,file)


