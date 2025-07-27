## Data Preprocessing 
### step 1: IMPORTING THE LIBRARIES <br/>

### step 2: IMPORTING THE DATASET   <br/>
you need to divide the data into two sets based on the dataset provided   <br/>
features set( X )  <br/>
target set (Y)
- [row, column] , a range in python includes the lower bound but excludes the upper bound :  EG: [ : , :3 ]   # all rows and columns like 0 , 1 and  2 will be used  

### Step 3: TAKING CARE OF MISSING DATA 
If the missing data is not much or does not participate as much in the analysis then: 
- Delete the missing values  or 
- Replace the missing values by the average or median or 
- Insert the most frequent value (categorical data) of all the values in the column in which the data is missing #except all the columns of matrix of features (X)  with numerical values , i.e. all numerical columns must  be included  

### Step 4: ENCODING CATEGORICAL DATA
- One Hot Encoding: A technique used to convert categorical data into a format that can be provided to machine learning algorithms to improve performance. In one-hot encoding, each unique category (or value) in a categorical feature is transformed into a binary vector
#so the country column would be turned into three columns because there are three different categories in the country column 
##### STEP 4.1: ENCODING THE INDEPENDENT VARIABLE (X’s)  Question: do independent variables need after encoding need to be a numpy array ? 
Answer: It is not strictly necessary but its often required or convenient . 
1. Machine learning libraries Expect Numpy arrays (scikit-learn)
2. Numpy arrays are memory efficient and computationally faster Han pandas Dataframes , or Python Lists
3. Compatibility with scalers( MinMaxScalers , StandardScaler)
4. Uniform Handling: If your data starts as a pandas data frame , converting it into numpy array ensures uniform handling of data across different stages of preprocessing or modelling   When its not necessary:
- 1. When using pandas data frame throughout
- 2. Custom Pipelines: If you have a pipeline that is specifically designed to handle pandas dataframe , there may not be the need to convert into lumpy array
##### STEP 4.2: ENCODING THE DEPENDENT VARIABLE (Y) 
dependent variable do not need to be Numpy array.  

### Step 5: SPLITTING THE DATA INTO TRAIN_TEST_SPLIT  Training set: 
train set: 
- x_train - matrix of feature of the training set ;  <br/> 
- y_train - dependent variable of the training set   <br/>
  
test set:
- x_test - matrix of feature of the test set ;  <br/> 
- y_test - dependent variable of the test set   <br/> 
### Step 6: FEATURE SCALING 

- we apply feature scaling after splitting the data into training set and test set.(to avoid information leakage/overfitting) or as the test set is suppose to be a brand new set only used for evaluation. 
- feature scaling = consists of scaling all your variables, all your features to make sure they all take value in the same scale 
- why do we need feature scaling ? to avoid some features to be dominated 

Feature scaling techniques:  
1. (Standardisation) Xstand = {x - mean(x)} / {standard deviation(x)}, this will put all the values of the feature btwn -3 & +3 , can be used in all scenarios 

 2. (Normalisation) Xnorm = { x - min(x)} / { max(x) - min(x) }, all the values of the feature  will be btwn 0 & 1 , when you have normalisation in most of your features   
dummy variables are these values 0.0 0.0 1.0 (that represent France in this scenario) which we obtained using one hot encoding .Feature scaling is not applied to dummy variables because they are already in a standardised format (0 or 1), which effectively represents categories without magnitude or scale, making scaling unnecessary.

for the test set , since it needs to be like a new data we will only apply transform methd , because the features of the test set need to be scaled by the same scaler that was used on the training set if we apply fit_transform methd on the X_test we will get a new scaler , which we dont want. 
