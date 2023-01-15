

#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import pickle


# In[29]:


from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve, accuracy_score
from sklearn.tree import DecisionTreeClassifier

pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)

def prediction(input_data):  
    prediction = model.predict(
        [input_data])
    print(prediction)
    return prediction

st.title("LinkedIn User Prediction")
st.markdown("This app was created to predict the likelihood of someone being a LinkedIn user. The following variables are used in the model. Use the fields below to check it out!")
age = st.slider("What is your age?", 1, 100,1)
educ2=st.selectbox("What is your education?",options=['Less than high school (Grades 1-8 or no formal schooling)' , 'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)' , 'High school graduate (Grade 12 with diploma or GED certificate)', 
                                        'Some college, no degree (includes some community college)','Two-year associate degree from a college or university',
                                        'Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)','Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)',
                                        'Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)'])

income=st.selectbox("What is your income",options=['Less than $10,000', '10 to under $20,000', '20 to under $30,000', '30 to under $40,000', '40 to under $50,000',
                                         '50 to under $75,000', '75 to under $100,000', '100 to under $150,000', '$150,000 or more'])
	
female=st.selectbox("What is your sex?",options=['Female','Male'])
married=st.selectbox("Are you married?",options=['No','Yes'])
parent=st.selectbox("Are you a parent?",options=['No','Yes'])

educ2 = 1 if educ2 == 'Less than high school (Grades 1-8 or no formal schooling)' else 0
educ2 = 2 if educ2 == 'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)' else 0
educ2 = 3 if educ2 == 'High school graduate (Grade 12 with diploma or GED certificate)' else 0
educ2 = 4 if educ2 == 'Some college, no degree (includes some community college)' else 0
educ2 = 5 if educ2 == 'Two-year associate degree from a college or university' else 0
educ2 = 6 if educ2 == 'Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)' else 0
educ2 = 7 if educ2 == 'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)' else 0
educ2 = 8 if educ2 == 'Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)' else 0

income = 1 if income == 'Less than $10,000' else 0
income = 2 if income == '10 to under $20,000' else 0
income = 3 if income == '20 to under $30,000' else 0
income = 4 if income == '30 to under $40,000' else 0
income = 5 if income == '40 to under $50,000' else 0
income = 6 if income == '50 to under $75,000' else 0
income = 7 if income == '75 to under $100,000' else 0
income = 8 if income == '100 to under $150,000' else 0
income = 9 if income == '$150,000 or more' else 0
female = 0 if female == 'Male' else 1
married = 0 if married == 'No' else 1
parent = 0 if parent == 'No' else 1

result=""
input_data = [age , educ2, income , female, married, parent]


predict_probability = model.predict_proba([input_data])
predicted_class = model.predict([input_data])

# In[37]:

#person=[82,3,8,1,1,1]
#predicted_class=model.predict([person])
#probs=model.predict_proba([person])

#st.write(f"Predicted class: {prediction}")
st.write(f"Probability that this person uses LinkedIn: {predict_probability[0][1]}")

if st.button("Predict"):
    result = prediction(input_data)
    st.success('The output is {}'.format(result))
