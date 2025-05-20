#IMPORTING LIBRARIES 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 



#IMPORTING THE DATASET 
dataset = pd.read_csv('Data.csv')
#create two new entities , 1st Matrix of features , 2nd dependent variable vector 
#[row, column], A range in python includes the lower bound but excludes the upperbound 
X= dataset.iloc[:, :3].values 
y = dataset.iloc[:, -1 ].values 



#TAKING CARE OF MISSING DATA 
#1. Ignore the observation by deleting it
#2. replace the missing value by the avg of all the values in the column in which the data is missing 

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') 
# you could replace it by avg or median or by the most frequent value(for categories)
imputer.fit(X[:, 1:3]) #expects all the coulmns of Matrix of features i.e. X with numerical values (all numerical value columns to be included)
X[:, 1:3] = imputer.transform(X[:, 1:3])
#print(X)




#ENCODE CATEGORICAL DATA 
# One-hot encoding: A technique used to convert categorical data into a format that can be provided to machine learning algorithms to improve performance. In one-hot encoding, each unique category (or value) in a categorical feature is transformed into a binary vector
#so the country culmn would be turned into three coulmns bcuz there are three different categories in the country coulmn 

#ENCODING THE INDEPENDENT VARIABLE 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder() ,[0])], remainder= 'passthrough')   # 1.transformers - where we specify what kind of transformation we want to do and on which indexes of the columns we want to transform   2.remainder 
X = np.array(ct.fit_transform(X)) 
#print(X)

#ENCODING THE DEPENDENT VARIABLE 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)  #dependent variable vector does not need to be a numpy array 




#SPLITTING THE DATA INTO TRAIN & TEST SET 
#INPUT FORMAT FOR SPLITING THE DATA INTO TRAIN & TEST SETS 

#we apply feature scaling after spliting the data into training set and test set.(to avoid information leakage/overfitting) or as the test set is suppose to be a brand new set only used for evaluation. 
#x_train - matrix of feature of the training set ; y_train - dependent variable of the training set ; 
#x_test - matrix of feature of the test set ; y_test - dependent variable of the test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X, y , test_size = 0.2 , random_state= 1)

print(X_train)
print(X_test)
print(y_train)
print(y_test)



#FEATURE SCALING 
#feature scaling = consists of scaling all your variables, all your features to make sure they all take value in the same scale 
#why do we need feature scaling ? to avoid some features to be dominated 
#Feature scaling techniques:  
# 1. (Standardisation) Xstand = {x - mean(x)} / {standard deviation(x)}, this will put all the values of the feature btwn -3 & +3 , can be used in all scenarios 
# 2. (Normalisation) Xnorm = { x - min(x)} / { max(x) - min(x) }, all the values of the feature  will be btwn 0 & 1 , when you have normalisation in most of your features 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#dummy variables are these values 0.0 0.0 1.0 (that represent france in this scenario) which we obtained using one hot encoding 
#Feature scaling is not applied to dummy variables because they are already in a standardized format (0 or 1), which effectively represents categories without magnitude or scale, making scaling unnecessary.
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
#for the test set , since it needs to be like a new data we will only apply transform methd , bcuz the features of the test set need to be scaled by the same scaler that was used on the training set 
#if we apply fit_transform methd on the X_test we will get a new scaler , which we dont want 
X_test[:, 3:] = sc.transform(X_test[:, 3:])






