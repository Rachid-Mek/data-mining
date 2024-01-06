import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 
from sklearn.ensemble import HistGradientBoostingRegressor

def load_data():
    df = pd.read_csv('Data/Dataset1.csv' ) # Load the dataset
    df['P'] = pd.to_numeric(df['P'],errors='coerce')  
 
    return df
# ----------------------------------------------------------------------------------------------------------------


 

def replace_missing_values(dataset , method):
    """Replace missing values """
    if method == 'mean':
        dataset.fillna(dataset.mean(), inplace=True)
    elif method == 'median':
        dataset.fillna(dataset.median(), inplace=True)
    elif method == 'mode':
        dataset.fillna(dataset.mode(), inplace=True)
    return dataset
# ----------------------------------------------------------------------------------------------------------------

def detect_outliers(data):
    """Detect outliers using the MAD method"""
    outliers = {}
    for col in data.columns:
        if data[col].dtype != 'object':  # Consider only numerical columns
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            col_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
            if not col_outliers.empty:
                outliers[col] = col_outliers.index.tolist()
    return outliers
# ----------------------------------------------------------------------------------------------------------------

def Replace_outliers(dataset,  treatment_method='mean'): # Function to handle outliers
    """Replace outliers using the MAD method"""
    cleaned_data = dataset.copy() # Create a copy of the dataset
    outliers = detect_outliers(dataset) # Get the outliers using the MAD method
    for col in outliers:
        if treatment_method == 'median':
            median = dataset[col].median()
            cleaned_data.loc[outliers[col], col] = median
        elif treatment_method == 'mean':
            mean = dataset[col].mean()
            cleaned_data.loc[outliers[col], col] = mean
    return cleaned_data # Return cleaned dataset
# ----------------------------------------------------------------------------------------------------------------
def Handle_outliers(dataset , treatment_method='mean'):
   

    dataset = Replace_outliers(dataset,  treatment_method='mean')  # Handle outliers
    return dataset
 


def predict_missing_values(dataset):
    """Predict missing values using a HistGradientBoostingRegressor for all columns."""
    
    for column in dataset.columns:
        # Skip columns with no missing values
        if dataset[column].isnull().sum() == 0:
            continue
        
        # Separate data into features and target
        features = dataset.dropna(subset=[column])
        target = features[column]
        features = features.drop(columns=[column])
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Train a machine learning model
        model = HistGradientBoostingRegressor()
        model.fit(X_train, y_train)
        
        # Predict missing values
        missing_data = dataset[dataset[column].isnull()]
        missing_features = missing_data.drop(columns=[column])
        predictions = model.predict(missing_features)
        
        # Fill in missing values with predicted values
        dataset.loc[dataset[column].isnull(), column] = predictions
    
    return dataset
# ----------------------------------------------------------------------------------------------------------------


 

def handle_outliers_regression(dataset, max_iterations=80):
    if detect_outliers(dataset) == {}:
        print('No outliers detected')
        return dataset  # No outliers detected, exit the loop

    outliers = detect_outliers(dataset)

    for col, col_outliers in outliers.items():
        # Separate data into features and target
        features = dataset.drop(index=col_outliers, errors='ignore')
        target = features[col]
        features = features.drop(columns=[col])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Train a linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict outliers
        outliers_data = dataset.loc[col_outliers]
        outliers_features = outliers_data.drop(columns=[col])

        predictions = model.predict(outliers_features)
        predictions = predictions.astype(dataset[col].dtype)

        # Replace outliers with predicted values
        dataset.loc[col_outliers, col] = predictions
 

    return dataset

    # normalize data using Log Transformation 
 
# ----------------------------------------------------------------------------------------------------------------
 
def Reduction_V(dataset ,corr):
    """Drop columns with correlation greater than corr"""
    corr_matrix = dataset.corr()  # Calculate the correlation matrix
    columns_to_drop = []  # Initialize an empty list to store columns to drop
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > corr:
                colname = corr_matrix.columns[i]  # Get the name of the column
                columns_to_drop.append(colname)  # Append the column to the list
    dataset.drop(columns_to_drop, axis=1, inplace=True)  # Drop the columns
    return dataset
def Reduction_H(dataset): 
    """Drop duplicate rows and columns"""
    # print('dataset shape before dropping duplicates:', dataset.shape)
    # Drop duplicate rows( Elimination des redondances horizontales) :
    dataset.drop_duplicates(inplace=True) # Drop duplicate rows inplace (in the same DataFrame)

    # Drop duplicate columns (Elimination des redondances verticales)
    dataset = dataset.T.drop_duplicates().T # Transpose, drop duplicates, transpose back
    # drop lines that are similar to each other with 60% similarity
    dataset = dataset.drop_duplicates(subset=dataset.columns, keep='first', inplace=False, ignore_index=False)
    
    # print('dataset shape after dropping duplicates:', dataset.shape)
    return dataset

def normalize_min_max (dataset ,collumn) : 
    min_value = dataset[collumn].min() # Get the minimum value of the column  
    max_value = dataset[collumn].max() # Get the maximum value of the column
    new_min = 0  # Define a new minimum value
    new_max = 1  # Define a new maximum value
    dataset[collumn] = (dataset[collumn] - min_value) / (max_value - min_value) * (new_max - new_min) + new_min # Apply the formula to each element of the column
    return dataset[collumn]  # Return the column with normalized values
def normalize_min_max_all_columns(dataset): 
    for col in dataset.columns[:-1]:
            dataset[col] = normalize_min_max(dataset ,col) # Normalize the column
    return dataset # Return the dataset with normalized columns
 
def reduction_variance(dataset, threshold=0.3):
    """Drop columns with variance less than threshold"""
    columns_to_drop = []  # Initialize an empty list to store columns to drop
    for col in dataset.columns:
        if dataset[col].dtype != 'object':  # Consider only numerical columns
            if dataset[col].var() < threshold:
                columns_to_drop.append(col)  # Append the column to the list
    dataset.drop(columns_to_drop, axis=1, inplace=True)  # Drop the columns
    return dataset
def Normalizer(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True) # Calculate the norms of the rows of X , keep the same dimensions , i.e. return a column vector  
    norms[norms == 0] = 1  # Avoid division by zero for zero-length vectors
    X_normalized = X / norms
    return X_normalized

def Preprocessing(corr=0.8 ,Norm=1):
    dataset = load_data() # Load the dataset

    # --------------------------- MISSING VALUES ----------------------------------------
    # dataset = replace_missing_values(dataset, 'mean') # Replace missing /values with the mean 
    dataset = predict_missing_values(dataset) # Replace missing values with predicted values 
    # -------------------------------------------------------------------

    # ------------------------- OUTLIERS ------------------------------------------
    # dataset = Handle_outliers(dataset , treatment_method='median') # Handle outliers  
    if Norm == 1 :
        dataset = normalize_min_max_all_columns(dataset)  
    elif Norm==2 :
        dataset = Normalizer(dataset)
 
    dataset = handle_outliers_regression(dataset)
    dataset = Reduction_H(dataset) # Drop duplicate rows and columns
    # -------------------------------------------------------------------
    dataset_before = dataset.copy() # Create a copy of the dataset before handling outliers
    dataset = Reduction_V(dataset, corr) # Drop columns with correlation greater than corr
    # dataset = reduction_variance(dataset ,var) #  Drop columns with variance less than var
    return dataset , dataset_before # Return the cleaned dataset and the dataset before  reducing the number of columns
 
def Preprocessing_1():
    dataset = load_data() # Load the dataset
    dataset = predict_missing_values(dataset) # Replace missing values with the mean
    #normalize data withoutout the last column
    dataset = normalize_min_max_all_columns(dataset)

    dataset = Handle_outliers(dataset) # Handle outliers
    dataset = Reduction_V(dataset, 0.9) # Drop columns with correlation greater than 0.9
    dataset = Reduction_H(dataset) # Drop duplicate rows and columns
    return dataset
# ----------------------------------------------------------------------------------------------------------------