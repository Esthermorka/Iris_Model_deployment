# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:55:34 2023

@author: Admin
"""

#importing dependencies
import numpy as np
import pandas as pd
from joblib import load
import streamlit as st

# loading in the data(The dumped model)
model = load('../model/Random_forest_classifier.joblib')


#Backend
def predictions(sepallength, sepalwidth, petallength, petalwidth):
    prediction = model.predict(np.array([[sepallength, sepalwidth, petallength, petalwidth]]))
    
    return prediction



# fuction to create the UI
def main():
    st.title('Iris flower model')
    
    
    

    sepallength = st.slider('Enter sepal length:', 4.3, 7.9, 5.9)
    sepalwidth = st.slider('Enter sepal width: ', 2.0, 4.4, 3.1)
    petallength = st.slider('Enter petal length: ',1.0, 6.9, 3.7)
    petalwidth = st.slider('Enter petal width: ', 0.1, 2.5, 1.2)
    
    
    button = st.button('predict')
    

    
    
    result = ''

    if(button):
        result = predictions(sepallength, sepalwidth, petallength, petalwidth)
        if result == 0:
            st.success('Setosa')
        if result == 1:
            st.success('Versicolor')
        else:
            st.success('Virginica')
    
    
if __name__ == '__main__':
    main()