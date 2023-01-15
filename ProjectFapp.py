#!/usr/bin/env python
# coding: utf-8



import streamlit as st
import pandas as pd
import numpy as np
import pickle
#import sklearn

# In[29]:


from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve, accuracy_score
from sklearn.tree import DecisionTreeClassifier


# **1. Read in the data, call the dataframe "s"  and check the dimensions of the dataframe**

# In[5]:

st.title("LinkedIn User Prediction")
st.markdown("This app was created to predict the likelihood of someone being a LinkedIn user. The following variables are used in the model. Enter the info and check the prediction!")

s=pd.read_csv('social_media_usage.csv')



# **2. define a function clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected**

# In[6]:


def clean_sm(x):
    x=np.where(x == 1,1,0)
    return x

# In[7]:

def clean_gender(x):
    x=np.where(x == 2,1,0)
    return x

# **3. Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.**

# In[8]:

ss=s.copy()
ss["sm_li"]=ss["web1h"].apply(clean_sm)
ss[ss["income"] > 9] = np.nan
ss[ss["educ2"] > 8] = np.nan
ss[ss["age"] > 98] = np.nan
ss[ss["par"] > 2]= np.nan
ss[ss["gender"] > 2] = np.nan
ss["female"]=ss["gender"].apply(clean_gender)
ss["married"]=ss["marital"].apply(clean_sm)
ss["parent"]=ss["par"].apply(clean_sm)

print(ss.columns)
ss=ss.dropna()

# In[9]:

ss=ss[ss.columns[ss.columns.isin(['income', 'educ2','age','female','parent','married','sm_li'])]]
#New Dataset 
print(ss)
ss.describe()


# Create the target vector
y = ss['sm_li']

# Create the feature set
X = ss.drop(columns=['sm_li'])


# In[22]:

X[['educ2','income']] = X[['educ2','income']].astype("category")

#print(X)

# In[24]:

X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,test_size=0.2, random_state=42)

print(X_train)
print(y_train)
print(y_test)
print(X_test)


# **6.	Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.**

# In[25]:

model=LogisticRegression(solver='liblinear',random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

from sklearn.metrics import ConfusionMatrixDisplay
y_pred=model.predict(X_test)
print(y_pred)

#pickle_out = open("model.pkl", "wb")
#pickle.dump(model, pickle_out)
#pickle_out.close()

# In[36]:

#Streamlit app
entry_age = st.slider("What is your age?", 1, 100,1)
entry_educ2=st.radio("What is your education?",options=['Less than high school (Grades 1-8 or no formal schooling)', 'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)', 'High school graduate (Grade 12 with diploma or GED certificate)', 
                                                        'Some college, no degree (includes some community college)','Two-year associate degree from a college or university',
                                                        'Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)', 'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)',
                                                        'Postgraduate or professional degree, including masters, doctorate, medical or law degree (e.g., MA, MS, PhD, MD, JD)'])


entry_income=st.radio("What is your income",options=['Less than $10,000', '10 to under $20,000', '20 to under $30,000', '30 to under $40,000', '40 to under $50,000',
                                                     '50 to under $75,000', '75 to under $100,000', '100 to under $150,000', '$150,000 or more'])

entry_female=st.selectbox("What is your sex?",options=['Female','Male'])
entry_married=st.selectbox("Are you married?",options=['No','Yes'])
entry_parent=st.selectbox("Are you a parent?",options=['No','Yes'])


if entry_educ2 == 'Less than high school (Grades 1-8 or no formal schooling)':
    entry_educ2 = 1
elif entry_educ2 == 'High school incomplete (Grades 9-11 or Grade 12 with NO diploma)':
    entry_educ2 = 2
elif entry_educ2 == 'High school graduate (Grade 12 with diploma or GED certificate)':
    entry_educ2 = 3
elif entry_educ2 == 'Some college, no degree (includes some community college)':
    entry_educ2 = 4
elif entry_educ2 == 'Two-year associate degree from a college or university':
    entry_educ2 = 5    
elif entry_educ2 == 'Four-year college or university degree/Bachelors degree (e.g., BS, BA, AB)':
    entry_educ2 = 6
elif entry_educ2 == 'Some postgraduate or professional schooling, no postgraduate degree (e.g. some graduate school)':
    entry_educ2 = 7
else: entry_educ2 = 8

if entry_income == 'Less than $10,000':
    entry_income = 1
elif entry_income == '10 to under $20,000':
    entry_income = 2
elif entry_income == '20 to under $30,000':
    entry_income = 3
elif entry_income == '30 to under $40,000':
    entry_income = 4
elif entry_income == '40 to under $50,000':
    entry_income = 5
elif entry_income == '50 to under $75,000':
    entry_income = 6
elif entry_income == '75 to under $100,000':
    entry_income = 7
elif entry_income == '100 to under $150,000':
    entry_income = 8   
else: entry_income = 9

entry_female = 0 if entry_female == 'Male' else 1
entry_married = 0 if entry_married == 'No' else 1
entry_parent = 0 if entry_parent == 'No' else 1


input_data = [entry_age, entry_educ2, entry_income, entry_female, entry_married, entry_parent]
#st.write(input_data)
predicted_class = model.predict([input_data])
predict_probability = model.predict_proba([input_data])

predict_probability = predict_probability[0][1]*100

# In[37]:

if predicted_class > 0:
   outcome=('The model predicts that you are likely to be LinkedIn user.')
else:
   outcome=('The model predicts that you are likely not a LinkedIn user.')

if st.button('Click to See Your Results'):
    st.subheader(outcome)
    st.caption(f'Based on the entries in the above questions, there is a **{predict_probability:.2f}%** probability that you are a Linkedin user.')
