
# coding: utf-8

# In[1]:

# HEADER FILES
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
matplotlib.style.use('ggplot')
get_ipython().magic('matplotlib inline')
# special matplotlib argument for improved plots
from matplotlib import rcParams


# In[2]:

df = pd.read_csv('test.csv') # read test data
df.set_index('Id', inplace=True) # sets the index corresponding to the "Id" column and removes that column
print (df.shape) # prints the shape of dataframe


# In[3]:

# HANDLING MISSING VALUES
df["MSZoning"].fillna(df["MSZoning"].value_counts().index[0], inplace=True)
df["BsmtFullBath"].fillna(df["BsmtFullBath"].median(), inplace=True)
df["BsmtHalfBath"].fillna(df["BsmtHalfBath"].median(), inplace=True)
df["GarageType"].fillna("NA", inplace=True)
df["GarageYrBlt"].fillna(df["GarageYrBlt"].median(), inplace=True)
df.loc[df["GarageType"]=='NA', 'GarageFinish'] = 'NA'
df["GarageFinish"].fillna(df["GarageFinish"].value_counts().index[0], inplace=True)
df["GarageCars"].fillna(df["GarageCars"].value_counts().index[0], inplace=True)
df["GarageArea"].fillna(df["GarageArea"].median(), inplace=True)
df.loc[df["MasVnrArea"].isnull(), 'MasVnrType'] = 'None'
df["MasVnrArea"].fillna(0, inplace=True)
df["MasVnrType"].fillna(df["MasVnrType"].value_counts().index[0], inplace=True)


# In[4]:

Y = pd.read_csv('gt.csv') # TARGET CLASS from gt.csv
Y.set_index('Id', inplace=True)
X = df.iloc[:, 0:len(df.columns)] # FEATURE DATAFRAME

#print (pd.isnull(X).sum())

for col in X.columns.values:
    if X[col].dtype == 'int64':
        X[col] = X[col].astype('float64')
    if X[col].dtype == object:
        X[col] = X[col].astype('category')

X["MSSubClass"] = X["MSSubClass"].astype('category') # The MS subclasses are actually categories

# Feature Scaling
for col in X.columns.values:
    if X[col].dtypes == 'float64':
        X[col] = (X[col] - X[col].min())/(X[col].max() - X[col].min()) # Min-Max Scaling of floating variables
        X[col] = (X[col] - X[col].mean())/X[col].std() # Standardized Scaling of floating variables
        


# In[5]:

# ONE HOT ENCODING FOR CATEGORICAL FEATURES
X = pd.get_dummies(X, columns = X.select_dtypes(include=['category']))

# extra features occuring in one hot encoded TRAIN data file 
headers = ['Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn',
       'HouseStyle_2.5Fin', 'SaleCondition_Abnorml', 'SaleCondition_AdjLand',
       'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal',
       'SaleCondition_Partial']
for header in headers:
    X[header] = 0
print (X.shape)


# In[6]:

# IMPORTING MODELS
logreg = joblib.load('logistic_regression.pkl')
knn = joblib.load('k_nearest_neighbors.pkl')
dtree = joblib.load('decision_tree.pkl')
rf = joblib.load('random_forest.pkl')


# In[7]:

Y_pred_logreg = logreg.predict(X)
Y_pred_knn = knn.predict(X)
Y_pred_dtree = dtree.predict(X) # MOST ACCURATE CLASSIFIER 
Y_pred_rf = rf.predict(X)


# In[8]:

# Writing out.csv corresponding to the most accurate model
outfile = open("out.csv","w")
outfile.write("Id,SaleStatus\n")
i = 1461
for y in Y_pred_dtree:
    outfile.write(str(i)+","+y+"\n")
    i += 1
outfile.close()


# In[9]:

print ('Logistic Regression Accuracy: ', logreg.score(X, Y))
print ('K Nearest Neighbor Accuracy: ', knn.score(X, Y))
print ('Decision Tree Accuracy: ', dtree.score(X, Y))
print ('Random Forest Accuracy: ', rf.score(X, Y))

