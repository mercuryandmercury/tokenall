# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 18:14:51 2022

@author: SARKAR
"""



import pandas as pd
import streamlit as st 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.metrics import mean_absolute_error







st.write("#####energy prediction ccpp")




def file_selector(self):
   file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
   if file is not None:
      data = pd.read_csv(file)
      return data
   else:
      st.text("Please upload a csv file")


# Standardize the feature data
X = data.loc[:, data.columns != self.chosen_target]
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X)
X = pd.DataFrame(scaler.transform(X))
X.columns = data.loc[:, data.columns != self.chosen_target].columns
y = data[self.chosen_target]




  