
# coding: utf-8

# In[238]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[239]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# In[240]:


train.head()


# In[241]:


test.head()


# In[242]:


train.shape


# In[243]:


test.shape


# In[244]:


train.describe()


# In[245]:


test.describe()


# In[246]:


train['Property_Area'].value_counts()


# In[247]:


train['ApplicantIncome'].hist(bins=50, color='Blue')
plt.show()


# In[248]:


train.boxplot(column='ApplicantIncome')
plt.show()


# In[249]:


train.boxplot(column='ApplicantIncome', by='Gender')
plt.show()


# In[250]:


train.boxplot(column='ApplicantIncome', by='Education')
plt.show()


# In[251]:


train['LoanAmount'].hist(bins=50 ,color='Green')
plt.show()


# In[252]:


train.boxplot(column='LoanAmount', by='Gender')
plt.show()


# In[253]:


train.boxplot(column='LoanAmount', by='Education')
plt.show()


# In[254]:


temp1 = train['Credit_History'].value_counts()
temp2 = train.pivot_table(values='Loan_Status', index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1, 'N':0}).mean())
print("Frequency table for Credit History.")
print(temp1)
print("Probability of getting Loan for each credit history class.")
print(temp2)


# In[255]:


fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax1.set_xlabel('Credit_History')
ax1.set_ylabel('Count of Applications')
ax1.set_title('Applicants by Credit_History')
temp1.plot(kind='bar')

'''ax2 = fig.add_subplot(122)
ax2.set_xlabel('Credit_History')
ax2.set_ylabel('Probablity of getting loan')
ax2.set_label('Probability of getting loan by credit history')
'''
temp2.plot(figsize=(4,4),title='Probability of getting loan by Credit_History',kind='bar')

plt.show()


# In[256]:


temp3 = pd.crosstab([train['Credit_History'],train['Gender']],train['Loan_Status'])
temp3.plot(kind='bar',stacked=True,color=['red','blue'],grid=False)
plt.show()


# In[257]:


print(train.isnull().any())


# In[258]:


train.apply(lambda x: sum(x.isnull()),axis=0)


# In[259]:


train.boxplot(column='LoanAmount',by=['Education','Self_Employed'])
plt.show()


# In[260]:


train['Self_Employed'].value_counts()


# In[261]:


train['Self_Employed'].fillna('No', inplace=True)


# In[262]:


table = train.pivot_table(values='LoanAmount',
                          index='Self_Employed',
                          columns='Education',
                          aggfunc=np.median)
table


# In[263]:


def fage(x):
    return table.loc[x['Self_Employed'],x['Education']]


# axis=0 means operations done column wise and axis=1 means row wise.

# In[264]:


train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()]
                          .apply(fage, axis=1), inplace=True)


# Treating Extreme values

# In[265]:


train['LoanAmount_log'] = np.log(train['LoanAmount'])
train['LoanAmount_log'].hist(bins=20,color='cyan')
plt.show()


# In[266]:


train['TotalIncome'] = train['ApplicantIncome'] + train['CoapplicantIncome']
train['TotalIncome_log'] = np.log(train['TotalIncome'])
train['TotalIncome_log'].hist(bins=20)
plt.show()


# In[296]:


train.isnull().any()


# In[268]:


train.dtypes


# In[298]:


train['Loan_Amount_Term'].value_counts()
train['Loan_Amount_Term'].fillna(360.0,inplace=True)


# Building the predictive model

# In[269]:


from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education',
          'Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    train[i] = le.fit_transform(train[i].astype('str'))
train.dtypes    


# In[270]:


train['Credit_History'].value_counts()
train['Credit_History'].fillna(1.0,inplace=True)


# In[271]:


#Importing models from scikit learn module
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# In[290]:


def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train1, test1 in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train1,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train1]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test1,:], data[outcome].iloc[test1]))
 
  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 


# In[292]:


#Logistic Regression
outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, train, predictor_var, outcome_var)


# In[293]:


predictor_var = ['Credit_History','Education','Married',
                'Self_Employed','Property_Area']
classification_model(model, train, predictor_var,outcome_var)


# In[294]:


#Decision Tree
model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, train, predictor_var, outcome_var)


# In[299]:


#different combination
predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model,train,predictor_var,outcome_var)


# In[301]:


#random forest
model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model,train,predictor_var,outcome_var)


# In[302]:


featimp = pd.Series(model.feature_importances_,index=predictor_var).sort_values(ascending=False)


# In[303]:


print(featimp)


# In[304]:


model = RandomForestClassifier(n_estimators=25,min_samples_split=25,max_depth=7,max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log',
                 'Credit_History','Dependents','Property_Area']
classification_model(model,train,predictor_var,outcome_var)

