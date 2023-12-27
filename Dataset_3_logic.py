from ast import literal_eval
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from itertools import combinations

# ---------------------------------------------------------------------------------------------------------
def load_dataset():
    dataset = pd.read_csv('Data/Dataset3.csv' , delimiter=',', quotechar='"')
    return dataset
# --------------------------------------------------------------------------------------------------------- 
def describe_data(data):
    return data.describe()
# ---------------------------------------------------------------------------------------------------------
def column_to_discretize():
    column_to_discretize = ['Temperature', 'Humidity', 'Rainfall']
    return column_to_discretize
# ---------------------------------------------------------------------------------------------------------
def Equal_frequency_discretize(dataset , column_name , q=0):
    n= len(dataset[column_name])
    # define the number of quantiles with a formula
    if q == 0:
        q = math.ceil(1+10/3*math.log10(n))
    dataset.sort_values(by=[column_name], inplace=True)
    dataset.reset_index(drop=True, inplace=True)
    bin_size = len(dataset) // q + 1
    discretized_column = [round(i // bin_size )for i in range(len(dataset))]
    dataset[f'{column_name}_E_F'] = discretized_column
    return dataset

# ---------------------------------------------------------------------------------------------------------
def Equal_width_discretize(dataset , column_name , num_bins=0):
    import warnings
    warnings.filterwarnings('ignore') # to ignore warnings

    # Define a list of columns to process
    columns_to_process = ['Temperature', 'Humidity', 'Rainfall']

    for i in range(len(dataset)): # for each row
        for column in columns_to_process: # for each column
            if type(dataset[column][i]) == str: # if the value is a string
                if ',' in dataset[column][i]: # if comma is used as decimal separator
                    dataset[column][i] = dataset[column][i].replace(',', '.') # replace comma with dot
                dataset[column][i] = float(dataset[column][i]) # convert to float
            else:
                dataset[column][i] = float(dataset[column][i]) # convert to float
    n= len(dataset[column_name]) # number of rows
    if num_bins == 0:
        num_bins =  1+10/3*math.log10(n) # number of bins
    min_value = dataset[column_name].min() # min value
    max_value = dataset[column_name].max()  # max value
    bin_width = (max_value - min_value) / num_bins  # bin width
    # print(f"Number of bins: {num_bins}")
    discretized_column = [round( (x - min_value) / bin_width ) for x in dataset[column_name]] # discretized column
    dataset[f'{column_name}_E_W'] = discretized_column # add discretized column to the dataframe
    return dataset
# ---------------------------------------------------------------------------------------------------------
def plot_classes(input_df, column= 'Temperature_E_W'):
    sns.countplot(x=column, data=input_df)
    plt.xlabel('classes')
    plt.ylabel('Count')
    plt.title('class distribution Temperature ')
    return plt
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# =========================================== APRIORI ==============================================================
# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
def frequency(transactions, itemset):
    count = 0
    for transaction in transactions:
        if itemset.issubset(transaction):
            count += 1
    return count
# ---------------------------------------------------------------------------------------------------------
def generate_candidates(frequent_itemsets, k):
    candidates = set()
    for i in range(len(frequent_itemsets) - 1):
        for j in range(i + 1, len(frequent_itemsets)):
            # Joining step
            new_candidate = frequent_itemsets[i][0].union(frequent_itemsets[j][0])
            # Pruning step
            if len(new_candidate) == k:
                candidates.add(tuple(sorted(new_candidate)))
    return [set(candidate) for candidate in candidates]
    #return [item for item in candidates if len(item)==k]
# ---------------------------------------------------------------------------------------------------------
def get_frequent_itemsets(transactions, min_support):
    # Get unique items
    unique_items = set(item for transaction in transactions for item in transaction)
    
    # List to save all itemsets that verify the min_support
    frequent_itemsets = []
    k = 1
    
    # First iteration candidates
    candidate_itemsets = [frozenset([item]) for item in unique_items]
    
    while True:
        # List to save each iteration's frequent itemsets with their support value
        frequent_itemsets_k = []
        
        # Calculate support
        for itemset in candidate_itemsets:
            support = frequency(transactions, itemset)
            
            if support >= min_support:
                frequent_itemsets_k.append((itemset, support))
                
        # No itemsets satisfy the min_support
        if not frequent_itemsets_k:
            break
        
        # Save the new itemsets
        frequent_itemsets.extend(frequent_itemsets_k)
        k += 1
        
        # Generate candidates for the next iteration
        candidate_itemsets = generate_candidates(frequent_itemsets_k, k)

    return frequent_itemsets
# ---------------------------------------------------------------------------------------------------------
# Modify the association_rules format to store rules as sets
def generate_association_rules(frequent_itemsets):
    association_rules = []
    for itemset, support in frequent_itemsets:
        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                consequent = itemset - set(antecedent)
                association_rules.append((set(antecedent), consequent, support))
    return association_rules
def association_rules(dataset ,min_support ):
    grouped_data = dataset.groupby('Temperature_E_W').agg({
        'Temperature': list,
        'Humidity': list,
        'Rainfall': list,
        'Soil': list,
        'Crop': list,
        'Fertilizer': list
    }).reset_index()
        #generer les transactions
    transactions = []
    for row in grouped_data.index:
        transactions.extend(list(zip(*grouped_data.iloc[row, 1:])))
        

    transactions = [set(transaction) for transaction in transactions]

    frequent_itemsets = get_frequent_itemsets(transactions, min_support) 
    association_rules = generate_association_rules(frequent_itemsets)
    # Convert sets to lists
    association_rules_list = [
        (list(item[0]), list(item[1]), item[2]) for item in association_rules
    ]

    # Create DataFrame
    association_rules_df = pd.DataFrame(association_rules_list, columns=['Antecedent', 'Consequent', 'Support'])

    # Explode the lists to separate rows
    association_rules_df = association_rules_df.explode('Antecedent').explode('Consequent')


    return association_rules_df
# ---------------------------------------------------------------------------------------------------------
def calculate_confidence(transaction_data, antecedent, consequent):
    # Nombre de transactions contenant l'antécédent et le conséquent
    support_A_and_B = sum(1 for transaction in transaction_data if antecedent.issubset(transaction) and consequent.issubset(transaction))

    # Nombre de transactions contenant l'antécédent
    support_A = sum(1 for transaction in transaction_data if antecedent.issubset(transaction))

    # Calculer la confiance
    confidence = support_A_and_B / support_A
    return confidence
# ---------------------------------------------------------------------------------------------------------
def calculate_lift(transactions, left, right):
    # Nombre de transactions contenant l'antécédent et le conséquent
    support_A_and_B = sum(1 for transaction in transactions if left.issubset(transaction) and right.issubset(transaction))

    # Nombre de transactions contenant l'antécédent
    support_A = sum(1 for transaction in transactions if left.issubset(transaction))

    # Nombre de transactions contenant le conséquent
    support_B = sum(1 for transaction in transactions if right.issubset(transaction))

    # Calculer la confiance
    confidence = support_A_and_B / support_A

    # Calculer le lift
    lift = confidence / support_B
    return lift
# ---------------------------------------------------------------------------------------------------------
def calculate_cosine(transactions ,left, right):
    # Nombre de transactions contenant l'antécédent et le conséquent
    support_A_and_B = sum(1 for transaction in transactions if left.issubset(transaction) and right.issubset(transaction))

    # Nombre de transactions contenant l'antécédent
    support_A = sum(1 for transaction in transactions if left.issubset(transaction))

    # Nombre de transactions contenant le conséquent
    support_B = sum(1 for transaction in transactions if right.issubset(transaction))

    # Calculer le cosine
    cosine = support_A_and_B / math.sqrt(support_A * support_B)
    return cosine
# ---------------------------------------------------------------------------------------------------------
def calculate_jaccard(transactions, left, right):
    # Nombre de transactions contenant l'antécédent et le conséquent
    support_A_and_B = sum(1 for transaction in transactions if left.issubset(transaction) and right.issubset(transaction))

    # Nombre de transactions contenant l'antécédent
    support_A = sum(1 for transaction in transactions if left.issubset(transaction))

    # Nombre de transactions contenant le conséquent
    support_B = sum(1 for transaction in transactions if right.issubset(transaction))

    # Calculer le jaccard
    jaccard = support_A_and_B / (support_A + support_B - support_A_and_B)
    return jaccard
# ---------------------------------------------------------------------------------------------------------
def Apriori(dataset , min_support, min_confidence ):
    # drop Temperature Humidity Rainfall columns
    dataset = dataset.drop(columns=['Temperature', 'Humidity', 'Rainfall' , 'Temperature_E_W', 'Humidity_E_W', 'Rainfall_E_W'])
    # Apply discretization
    dataset['Temperature_E_F'] = dataset['Temperature_E_F'].replace(0,'Low')
    dataset['Temperature_E_F'] = dataset['Temperature_E_F'].replace(1,'Medium')
    dataset['Temperature_E_F'] = dataset['Temperature_E_F'].replace(2,'High')

    print(dataset)
    transactions = dataset[['Temperature_E_F', 'Soil', 'Crop', 'Fertilizer']].values.tolist()
   
    frequent_itemsets = get_frequent_itemsets(transactions, min_support) 
    association_rules = generate_association_rules(frequent_itemsets)
    # Convert association rules to a dataframe
    association_rules_df = pd.DataFrame(association_rules, columns=['Antecedent','consequent', 'support'])
   
    association_rules_df['confidence'] = association_rules_df.apply(lambda row: calculate_confidence(transactions, row['Antecedent'], row['consequent']), axis=1) 
    association_rules_df = association_rules_df[association_rules_df['confidence'] >= min_confidence].reset_index(drop=True)  
   
    association_rules_df['lift'] = association_rules_df.apply(lambda row: calculate_lift(transactions,row['Antecedent'], row['consequent']), axis=1)
    association_rules_df = association_rules_df[association_rules_df['confidence'] >= min_confidence].reset_index(drop=True)
 
    association_rules_df['cosine'] = association_rules_df.apply(lambda row: calculate_cosine(transactions,row['Antecedent'], row['consequent']), axis=1)
    association_rules_df['jaccard'] = association_rules_df.apply(lambda row: calculate_jaccard(transactions ,row['Antecedent'], row['consequent']), axis=1)
    
    return association_rules_df

# ---------------------------------------------------------------------------------------------------------
def Apply_discritization(dataset):
    column_to_discretize = ['Temperature', 'Humidity', 'Rainfall']
    for column in column_to_discretize:
        dataset = Equal_frequency_discretize(dataset , column ,3)
        dataset = Equal_width_discretize(dataset , column)
    return dataset
# ---------------------------------------------------------------------------------------------------------
 
    
def Temperature_class(temp):
    if temp < 24.88:
        return 'Low'
    elif temp < 26.77:
        return 'Medium'
    else:
        return 'High'


def Predict(dataset ,Temperature,  Soil , Crop , Fertilizer):
    rules = Apriori(dataset , 10 , 0.5)
    Antecedent = []
    Consequent = []
    if Soil != '':
        Antecedent.append(Soil)
    if Crop != '':
        Antecedent.append(Crop)
    if Fertilizer != '':
        Antecedent.append(Fertilizer)
    if Temperature != '':
        Temperature = Temperature_class(Temperature)
        Antecedent.append(Temperature)
    print(rules)
    print(Antecedent)
    print(Consequent)
    for i in range(len(rules)):
        print(list(rules['Antecedent'][i]))
        if set(Antecedent) == set(rules['Antecedent'][i]) :
            Consequent.append(rules['consequent'][i])
    if len(Consequent) == 0:
        return 'No rules found'
    return Consequent