

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

def prediction(age, educ2, income, female, married, parent):  
    prediction = model.predict(
        [[age, educ2, income, female, married, parent]])
    print(prediction)
    return prediction

st.title("LinkedIn User Prediction")
st.markdown("This app was created to predict the likelihood of someone being a LinkedIn user. The following variables are used in the model. Use the fields below to check it out!")
age = st.slider("What is your age?", 1, 100,1)
educ2=st.selectbox("What is your education?",options=[1:'Less than high school (Grades 1-8 or no formal schooling)' , 2:'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)' , 3:'High school graduate (Grade 12 with diploma or GED certificate)', 
                                        4:'Some college, no degree (includes some community college)',5:'Two-year associate degree from a college or university',
                                        6:'Four-year college or university degree/Bachelor’s degree (e.g., BS, BA, AB)',7:'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)',
                                        8:'Postgraduate or professional degree, including master’s, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)'])

income=st.selectbox("What is your income",options=[1:'Less than $10,000', 2:'10 to under $20,000', 3:'20 to under $30,000', 4:'30 to under $40,000', 5:'40 to under $50,000',
                                         6:'50 to under $75,000', 7:'75 to under $100,000', 8:'100 to under $150,000', 9:'$150,000 or more'])
	
female=st.selectbox("What is your sex?",options=[1:'Female',0:'Male'])
married=st.selectbox("Are you married?",options=[0:'No',1:'Yes'])
parent=st.selectbox("Are you a parent?",options=[0:'No',1:'Yes'])

result=""
#input_data = [age , educ2, income , female, married, parent]


#predict_probability = model.predict_proba([prediction])
#predicted_class = model.predict([input_data])

# In[37]:

#person=[82,3,8,1,1,1]
#predicted_class=model.predict([person])
#probs=model.predict_proba([person])

#st.write(f"Predicted class: {prediction}")
#st.write(f"Probability that this person uses LinkedIn: {predict_probability[0][1]}/100")

if st.button("Predict"):
    result = prediction(age , educ2, income , female, married, parent)
    st.success('The output is {}'.format(result))
