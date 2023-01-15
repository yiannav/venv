#!/usr/bin/env python
# coding: utf-8


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
import pickle


# In[29]:


from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve, accuracy_score
from sklearn.tree import DecisionTreeClassifier


# **1. Read in the data, call the dataframe "s"  and check the dimensions of the dataframe**

# In[5]:

st.title("LinkedIn Prediction")
s=pd.read_csv('social_media_usage.csv')
print(type(s))
print(s.dtypes)
print(s.size)
print(s.shape)
print(s.ndim)
print(s.head)
s.describe()
print(s.columns)


# **2. define a function clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected**

# In[6]:


def clean_sm(x):
    x=np.where(x == 1,1,0)
    return x


# pass column names in the columns parameter 
df = pd.DataFrame({'Names':["David", "Evan", "Helena","Michael"],
                  'Age':[23,27,25,25]})

df['new_col'] = df['Age'].apply(clean_sm)

print(df)


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


# In[10]:


pd.crosstab(ss["income"], columns="count", normalize=True)


# In[11]:


pd.crosstab(ss["income"], ss["sm_li"])


# In[12]:


pd.crosstab(ss["educ2"], columns="count", normalize=True)


# In[13]:


ss.groupby('sm_li').mean().plot.bar()
plt.show()


# In[14]:


import seaborn
sns.pairplot(ss, hue='sm_li')
plt.show()


# In[15]:


pd.crosstab(ss["parent"], columns="count", normalize=True)


# In[16]:


pd.crosstab(ss["married"], columns="count", normalize=True)


# In[17]:


pd.crosstab(ss["female"], columns="count", normalize=True)


# In[18]:


age_hist=plt.hist(ss["age"])
print(age_hist)
ss.age.describe()


# In[19]:


sns.boxplot(x=ss["age"])


# In[20]:


sns.catplot(data=ss, x="sm_li", y="age", kind="box")


# **4.	Create a target vector (y) and feature set (X)**
#                  

# In[21]:


# Create the target vector
y = ss['sm_li']

# Create the feature set
X = ss.drop(columns=['sm_li'])
print(y)
print(X)


# In[22]:


X[['educ2','income']] = X[['educ2','income']].astype("category")

#X=pd.get_dummies(X, columns = ['educ2', 'income'], drop_first=False)


#print(X)


# In[23]:


#normalize
#from sklearn.preprocessing import MinMaxScaler
#scaler=MinMaxScaler()
#X_scaled=scaler.fit_transform(X)
#X=pd.DataFrame(X_scaled, columns=X.columns)

#print(X)


# **Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning** x includes the features (predictors). y is the target variable (dependent) use of LinkedIn (1=yes, 0=No), x_train is the 80% of the x dataframe that is going to be used to train the model and x_test includes 20% of the x dataframe that will be used to test the model. The y_train is the 80% of the target variable Y and will be used for training while y_test is the 20% that is set aside for testing. 

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
pickle.dump(model, open('final_model.sav', 'wb'))


# **7.	Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.** The model's score is .66. In terms of the confusion matrix the numbers indicate the following: The 108 value (yellow box) are the values that are true negatives - they are actually negative and predicted as such. The 28 value (purple box) are the false negatives, values that are positive but predicted as negative. The 57 value (blue box top) represents the false positives, that is values that are actually negative but predicted as positive. The 58 value (blue box bottom) are true positives, values that are positive and predicted as such. 

# In[30]:


from sklearn.metrics import ConfusionMatrixDisplay
y_pred=model.predict(X_test)
confusion_df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print(confusion_df)
score = accuracy_score(y_test,y_pred)
print(score)
outputconf=confusion_matrix(y_test,y_pred)
print(outputconf)
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.show()


# In[31]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# 8.	Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[32]:


df = pd.DataFrame(outputconf, columns = pd.MultiIndex.from_tuples([('Predicted','No LinkedIn'),('Predicted', 'Yes LinkedIn')]),
                 index=pd.MultiIndex.from_tuples([('Actual','No LinkedIn'),('Actual','Yes LinkedIn')]))
df


# **9. Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.**
# 
# + Precision indicates the proportion of positive predicted values that are correct in the observed/actual data. The ratio is primarily influenced by the number of false positives (the values predicted that they are positive but are actually negative) given that the formula is TP/TP + FP. The higher the FP values the lower the ratio. 
# + *For example, if the model was to predict disease diagnosis such as MS, we would want to err on the side of caution and would prefer a high precision metric given that within this context, we would prefer to predict that someone has the disease and be wrong rather than predicting they do not have it and it turns out that the do indeed have MS. The health care system has more than one way to diagnose a serious illness and a high precision score would indicate lower False-Positives.*    
# + Recall is the metric that is primarily influenced by false negatives (the values predicted as negative but are actually positive). The higher the FN values the lower the ratio given that the formula is TP/TP+FN. The recall metric is also referred to as the sensitivity score as it is sensitive to identifying positive values. 
# + *For example, if the model was to predict whether Argentina would win the world cup, a high recall score would indicate that we are less likely to have false negatives; and, if there is any gambling involved, one would want to rely on a high recall score which indicates that the model is good at identifying positive results.*
# + The F1 score is a combination of Precision and Recall and can be used to get a better sense of the balance between them. Since F1 is calculated using the harmonic mean of precision and recall, it might be a better metric to rely on for datasets that have inbalanced cases.
# + *The F1 score might be a better metric for models predicting sleep patterns from various wearable devices. Given that the variability in devices and their actual use, the F1 metric will give us a more balanced score for predicting whether one has a good night sleep or not.*
# 

# In[33]:


import math
#TP/ TP +FP
precision= 60/ (60 + 58)
print(precision)

# TP/ TP + FN
recall= 60 / (60  + 24)
print(recall)

F1_score=2 * (precision*recall) / (precision + recall)
print(F1_score)


# the classification report sklearn
print(classification_report(y_test, y_pred))



# **10.Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?**

# In[34]:


new_data=pd.DataFrame({
    "age":[42,82],
    "educ2":[7,7],
    "income": [8, 8],
    "female":[1,1],
    "married":[1,1],
    "parent":[0,0]    
})


# In[35]:


new_data


# In[36]:


new_data["prediction_linkedIn"]=model.predict(new_data)
new_data


# In[37]:


person=[42,7,8,1,1,0]
predicted_class=model.predict([person])
probs=model.predict_proba([person])


# In[38]:


print(f"Predicted class: {predicted_class[0]}")
print(f"Probability that this person uses LinkedIn: {probs[0][1]}")


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


input_data = [age , educ2, income , female, married, parent]
predict_probability = model.predict_proba([input_data])
predicted_class = model.predict([input_data])


# In[37]:

#person=[82,3,8,1,1,1]
#predicted_class=model.predict([person])
#probs=model.predict_proba([person])

st.write(f"Predicted class: {predicted_class[0]}")
st.write(f"Probability that this person uses LinkedIn: {predict_probability[0][1]}")
