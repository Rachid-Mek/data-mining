 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore') # to ignore warnings

def import_dataset(choice):
    if choice == 1:
        dataset = pd.read_csv('Dataset1.csv' , delimiter=',', quotechar='"')
        dataset['P'] = pd.to_numeric(dataset['P'],errors='coerce')
    elif choice == 2:
        dataset = pd.read_csv('Dataset2.csv' , delimiter=',', quotechar='"')
    elif choice == 3:
        dataset = pd.read_csv('Dataset3..csv' , delimiter=',', quotechar='"') # Load the data
    return dataset 
 

# Fournir une description globale du dataset.

def Description(dataset):
    print("Description du dataset :")
    print(dataset.describe())
    print("Description de chaque attribut du dataset :")
    print(dataset.info())
    print("Nombre de valeurs manquantes :")
    print(dataset.isnull().sum())
    print("Nombre de valeurs dupliquées :")
    print(dataset.duplicated().sum())
    print("Nombre de valeurs uniques :")
    print(dataset.nunique())
    print("Valeurs uniques de chaque attribut :")
    for col in dataset.columns:
        print(col, dataset[col].unique())
    print("Valeurs manquantes :")
    print(dataset.isna().sum())


def Description_table(dataset):
    Desccription_table = pd.DataFrame({
        'Description' : [  'Number of rows','Number of columns' ,
        'Number of missing values ', 'Number of duplicate values', 'Number of unique values'
            ],
        'Valeurs' : [ dataset.shape[0],dataset.shape[1] ,
        dataset.isnull().count().sum(), dataset.duplicated().sum(),  dataset.nunique().sum()  ]

    })
    return Desccription_table
def find_min_max_mode(dataset):
    """Find the minimum, maximum and mode of a dataset"""
    # Finding minimum value
    min_value = dataset[0]
    for value in dataset[1:]:
        if value < min_value:
            min_value = value
    
    # Finding maximum value
    max_value = dataset[0]
    for value in dataset[1:]:
        if value > max_value:
            max_value = value
    
    # Finding mode
    frequency = {}
    for value in dataset:
        if value in frequency:
            frequency[value] += 1
        else:
            frequency[value] = 1
    
    mode = None
    max_frequency = 0
    for value, freq in frequency.items():
        if freq > max_frequency:
            max_frequency = freq
            mode = value
    
    return min_value, max_value, mode
 

def Description_measures(dataset):
    """Calculate the measures of central tendency and dispersion for each column in the dataset"""
    # Fournir une description de chaque attribut du dataset.
    Description_measures = {'Measures' :[  'Data types'  , 'Unique values for each attribute' ,
    'Number of duplicates' , 'Number of missing values' , "Sum of values" , "min value" , "max_value", "Mode" ]}

    for col in dataset.columns[0:]:
        Description_measures[col]=[

        dataset.dtypes[col],
        dataset[col].nunique() , 
        dataset[col].duplicated().sum() ,
        dataset[col].isnull().sum(),
        round( dataset[col].sum() ,2),
        find_min_max_mode(dataset[col])[0],
        find_min_max_mode(dataset[col])[1],
        find_min_max_mode(dataset[col])[2]
        ]
    Description_measures = pd.DataFrame(Description_measures)
    return Description_measures




 
# Calculer les mesures de tendance centrale et de dispersion pour chaque attribut du dataset.
def Calculate_Measure(dataset):
    """Calculate the measures of central tendency and dispersion for each column in the dataset"""
    # Create an empty dictionary to store the measures
    measures = {'Mesure': ['Moyenne', 'Médiane', 'Variance', 'Q0:Min', 'Q1', 'Q2', 'Q3', 'Q4:Max' , 'Ecart-type', 'Skewness', 'Kurtosis'  ]}

    # Loop through each column in the dataset
    for col in dataset.columns[:]:
        # Calculate the required measures for each column and append to the dictionary
        measures[col] = [
            dataset[col].mean().round(2),
            round(dataset[col].median(), 2),
            round(dataset[col].var(), 2),
            sorted(dataset[col])[0],
            dataset[col].quantile(0.25),
            dataset[col].quantile(0.5),
            dataset[col].quantile(0.75),
            sorted(dataset[col])[-1],
            round(dataset[col].std(), 2),
            round(dataset[col].skew(), 2),
            dataset[col].kurtosis(),
        ]

    # Create a DataFrame from the dictionary
    Mesure_tendance_centrale = pd.DataFrame(measures)

    return Mesure_tendance_centrale
 
def create_box_plot(dataset, column):
    """Create a boxplot for a column in the dataset"""
    dataset.boxplot(column=column)
    plt.title(column)
    return plt

def create_box_plots(dataset):
    """Create boxplots for all columns in the dataset"""
    num_cols = len(dataset.columns)
    num_rows = num_cols // 3 + (num_cols % 3 > 0)  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))  # Create subplots

    for i, col in enumerate(dataset.columns):
        ax = axes[i // 3, i % 3] if num_rows > 1 else axes[i % 3]  # Get the current axis
        dataset.boxplot(column=col, ax=ax)
        ax.set_title(col)  # Set the title of the subplot

    plt.tight_layout()
    plt.show()



def create_histograms(dataset):
    """Create histograms for all columns in the dataset"""
    num_cols = len(dataset.columns[:])
    num_rows = num_cols // 3 + (num_cols % 3 > 0)  # Calculate the number of rows needed

    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))  # Create subplots

    for i, col in enumerate(dataset.columns[:]):
        ax = axes[i // 3, i % 3] if num_rows > 1 else axes[i % 3]  # Get the current axis
        dataset[col].plot(kind='hist', ax=ax, title=col)

    plt.tight_layout()
    plt.show()

def create_histogram(dataset, column):
    """Create a histogram for a column in the dataset"""
    plt.hist(dataset[column])
    plt.title(column)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    return plt


# Construire et afficher des diagrammes de dispersion des données et en déduire les  corrélations entre les propriétés du sol
def create_scatter_plots(dataset):
    """Create scatter plots for all columns in the dataset"""
    for i , col1 in enumerate(dataset.columns[:13]):
        for j , col2 in enumerate(dataset.columns[:13]):
            if dataset[col1].corr(dataset[col2]) > 0.4 and col1 != col2 :
                plt.scatter(dataset[col1], dataset[col2])
                plt.title(col1+' vs '+col2)
                plt.xlabel(col1)
                plt.ylabel(col2)
                plt.show()
 
def create_scatter_plot(data, x_column, y_column) :
    """Create a scatter plot for two columns in the dataset"""
    plt.scatter(data[x_column], data[y_column])
    plt.title(x_column + ' vs ' + y_column)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    return plt
 


# calculate matrice de correlation with heatmap
def calculate_correlation(dataset):
    """Calculate the correlation matrix for the dataset"""
    corr_matrix = dataset.corr()  # Calculate the correlation matrix
    plt.figure(figsize=(15, 15))
    sns.heatmap(corr_matrix, annot=True)
    plt.show()


 
# Choix de la méthode de remplacement des valeurs manquantes :
# Remplacer les valeurs manquantes par la moyenne de la colonne
def replace_missing_values(dataset , method):
    """Replace missing values """
    if method == 'mean':
        dataset.fillna(dataset.mean(), inplace=True)
    elif method == 'median':
        dataset.fillna(dataset.median(), inplace=True)
    elif method == 'mode':
        dataset.fillna(dataset.mode(), inplace=True)
    return dataset
 
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

 
# Choix de la méthode de traitement des valeurs aberrantes
# Utilisation de techniques robustes  : MAD (Median Absolute Deviation)

def Replace_outliers(dataset,  treatment_method='median'): # Function to handle outliers
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
        # elif treatment_method == 'remove':
        #     # Remove the outliers
        #     for index in outliers[col]:
        #         cleaned_data.drop(index, inplace=True)
    return cleaned_data # Return cleaned dataset

def Handle_outliers(dataset , treatment_method='median'):
    while detect_outliers(dataset) != {}:
        dataset = Replace_outliers(dataset,  treatment_method='median')  # Handle outliers
        dataset = Replace_outliers(dataset,  treatment_method='mean')  # Handle outliers
    return dataset

def Reduction_H(dataset ,corr):
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


# Réduction des données (élimination des redondances) horizontales / verticales.
def Reduction_V(dataset): 
    """Drop duplicate rows and columns"""
    # print('dataset shape before dropping duplicates:', dataset.shape)
    # Drop duplicate rows( Elimination des redondances horizontales) :
    dataset.drop_duplicates(inplace=True) # Drop duplicate rows inplace (in the same DataFrame)

    # Drop duplicate columns (Elimination des redondances verticales)
    dataset = dataset.T.drop_duplicates().T # Transpose, drop duplicates, transpose back

    # print('dataset shape after dropping duplicates:', dataset.shape)
    return dataset
 
# Normalisation des données (Min-Max)
def normalize_min_max (dataset , collumn) :
    """Normalize a column using the min-max method"""
    min_value = dataset[collumn].min() # Get the minimum value of the column
    max_value = dataset[collumn].max() # Get the maximum value of the column
    new_min = 0  # Define a new minimum value
    new_max = 1  # Define a new maximum value
    dataset[collumn] = (dataset[collumn] - min_value) / (max_value - min_value) * (new_max - new_min) + new_min # Apply the formula to each element of the column
    return dataset[collumn]  # Return the column with normalized values

def normalize_min_max_dataset(dataset):
    """Normalize all columns in the dataset using the min-max method"""
    dataset_norm_min_max = dataset.copy() # Create a copy of the dataset

    for col in dataset_norm_min_max.columns[:]: # Loop through each column in the dataset
        dataset_norm_min_max[col] = normalize_min_max(dataset_norm_min_max , col) # Normalize each column using the function above
    return dataset_norm_min_max # Return the normalized dataset
 

#  Z-score
# Z-score normalization is a method that re-scales based on mean and standard deviation
def normalize_z_score (dataset ,collumn) :
    """Normalize a column using the z-score method"""
    mean = dataset[collumn].mean() # Calculate the mean
    std = dataset[collumn].std() # Calculate the standard deviation
    dataset[collumn] = (dataset[collumn] - mean) / std # Apply the normalization formula
    return dataset[collumn] # Return the normalized column
def normalize_z_score_dataset(dataset):
    dataset_norm_z_score = dataset.copy() # Create a copy of the dataset

    for col in dataset_norm_z_score.columns[:]: # Loop through all columns in the dataset
        dataset_norm_z_score[col] = normalize_z_score(dataset_norm_z_score , col) # Normalize each column
    return dataset_norm_z_score # Return the normalized dataset









# ----------------------------PART 3 ----------------------------------------------------------------------------------
 
def equal_frequency_discretize(data , column_name , q=0):
    n= len(data[column_name])
    # define the number of quantiles with a formula if q is not provided
    if q == 0:
        q = math.ceil(1+10/3*math.log10(n)) # Sturges' formula
    print(f"Number of quantiles: {q}") # Print the number of quantiles
    data.sort_values(by=[column_name], inplace=True) # Sort the data by the column to discretize
    data.reset_index(drop=True, inplace=True)  # Reset the index
    bin_size = len(data) // q # Calculate the size of each bin
    discretized_column = [round(i // bin_size )for i in range(len(data))]  # Create a list of discretized values
    data[f'{column_name}_E_F'] = discretized_column # Add the discretized column to the DataFrame
    return data # Return the DataFrame

def Apply_equal_frequency_discretize(dataset , column_name , q):
    dataset = equal_frequency_discretize(dataset , column_name , q)  # Discretize the 'Temperature' column into 10 bins
    return dataset  
 

def correct_type_missmuch(data):
    # Define a list of columns to process
    columns_to_process = ['Temperature', 'Humidity', 'Rainfall']

    for i in range(len(data)): # for each row
        for column in columns_to_process: # for each column
            if type(data[column][i]) == str: # if the value is a string
                if ',' in data[column][i]: # if comma is used as decimal separator
                    data[column][i] = data[column][i].replace(',', '.') # replace comma with dot
                data[column][i] = float(data[column][i]) # convert to float
            else:
                data[column][i] = float(data[column][i]) # convert to float
    return data


def equal_width_discretize(data , column_name):
    n= len(data[column_name]) # number of rows
    num_bins =  1+10/3*math.log10(n) # number of bins
    min_value = data[column_name].min() # min value
    max_value = data[column_name].max()  # max value
    bin_width = (max_value - min_value) / num_bins  # bin width

    discretized_column = [round( (x - min_value) / bin_width ) for x in data[column_name]] # discretized column
    data[f'{column_name}_E_W'] = discretized_column # add discretized column to the dataframe
    return data

def Apply_equal_width_discretize(dataset , column_name):
    dataset = equal_width_discretize(dataset , column_name)  # Discretize the 'Temperature' column into 10 bins
    return dataset
 