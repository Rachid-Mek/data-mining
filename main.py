import time
import numpy as np
import pandas as pd
from Prep_dataset1 import Preprocessing_1
from Suprevised_Algo import *
from metrics import *
from K_means import K_MEANS
from DBSCAN import DBSCAN_


def custom_train_test_split(dataset, test_size=0.2, random_state=None):
    '''
    Splits a dataset into training and test sets.
    parametres:
    -----------
    
    dataset: The dataset to be split
    test_size: The proportion of the dataset to be included in the test set
    random_state: Random state for reproducibility
    returns:
    -------
    train_set: The training set
    test_set: The test set
    '''
    np.random.seed(random_state)  # Pour la reproductibilité du mélange
    
    # nombre de classes dans le dataset
    num_classes = len(np.unique(dataset[:, -1]))
    train_set = np.empty((0, dataset.shape[1]))
    test_set = np.empty((0, dataset.shape[1]))

    for i in range(0, num_classes):
        indices_classe = np.where(dataset[:, -1] == i)[0]
        np.random.shuffle(indices_classe)
        
        nb_test_samples = int(test_size * len(indices_classe))
        test_indices = indices_classe[:nb_test_samples]
        train_indices = indices_classe[nb_test_samples:]
        
        train_set = np.vstack((train_set, dataset[train_indices, :]))
        test_set = np.vstack((test_set, dataset[test_indices, :]))

    np.random.shuffle(train_set)
    np.random.shuffle(test_set)

    return train_set, test_set

def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

def manhattan_distance(x1, x2):
    distance = np.sum(np.abs(x1-x2))
    return distance

def cosine_distance(x1, x2):
    distance = 1 - np.dot(x1, x2) / (np.sqrt(np.dot(x1, x1)) * np.sqrt(np.dot(x2, x2)))
    return distance
#-----------------------------------------KNN-----------------------------------------#
def execute_knn(k,distance_function='Euclidean'):
    dataset = Preprocessing_1()
    train_set , test_set = custom_train_test_split(dataset.values, test_size=0.2, random_state=0)
    train_set = pd.DataFrame(train_set, columns=dataset.columns)
    test_set = pd.DataFrame(test_set, columns=dataset.columns)

    x_train = train_set.iloc[:, :-1]
    y_train = train_set.iloc[:, -1]
    x_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1]

    if distance_function == 'Euclidean':
        distance_function = euclidean_distance
    elif distance_function == 'Manhattan':
        distance_function = manhattan_distance
    elif distance_function == 'Cosine':
        distance_function = cosine_distance
    
    # Create an instance of the Knn class
    knn_classifier = Knn(k=k, distance_function=distance_function)

    # Fit the model on the training set
    knn_classifier.fit(np.array(x_train),np.array(y_train))

    knn_start = time.time()
    # Predict on the test set
    y_pred = knn_classifier.predict(np.array(x_test))

    knn_end = time.time()
    Knn_time = knn_end - knn_start
    
    y_test = y_test.tolist()
    y_pred = [int(i) for i in y_pred]
    y_test = [int(i) for i in y_test]
    
    # Compute the confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred, len(np.unique(y_test)))
    fig = plot_confusion_matrix(conf_mat, classes=['Class 0', 'Class 1','class 2'])

    # Compute the metrics
    knn_accuracy = EXACTITUDE(conf_mat)
    knn_specificity = SPECIFICITE(conf_mat)
    prec , knn_precision = PRECISION(conf_mat)
    rec , knn_recall = RAPPEL(conf_mat)
    knn_f1_score = F1_score(conf_mat)
    df_metrics = pd.DataFrame({'Accuracy': knn_accuracy, 'Specificity': knn_specificity, 'Precision': knn_precision, 'Recall': knn_recall, 'F1_score': knn_f1_score, 'Execution time':Knn_time}, index=[0])
    
    return fig, conf_mat, df_metrics , knn_classifier

#-----------------------------------------Decision_tree-----------------------------------------#
def execute_Dt(min_samples_split, max_depth, n_features = None):
    dataset= Preprocessing_1()
    train_set , test_set = custom_train_test_split(dataset.values, test_size=0.2, random_state=0)

    train_set = pd.DataFrame(train_set, columns=dataset.columns)
    test_set = pd.DataFrame(test_set, columns=dataset.columns)

    # Transforming the train and test sets into lists
    x_train = train_set.iloc[:, :-1]
    y_train = train_set.iloc[:, -1]
    x_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1]

    # Create an instance of the Decision_Tree class
    decision_tree = DecisionTree(min_samples_split=min_samples_split, max_depth=max_depth, n_features=n_features)

    # Fit the model on the training set
    decision_tree.fit(np.array(x_train), np.array(y_train))

    decision_tree_start = time.time()
    # Predict on the test set
    y_pred_dt = decision_tree.predict(np.array(x_test))
    decision_tree_end = time.time()
    Decision_Tree_time = decision_tree_end - decision_tree_start

    y_test = [int(i) for i in y_test]
    y_pred_dt = [int(i) for i in y_pred_dt]

    # Compute the confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred_dt, len(np.unique(y_test)))
    fig = plot_confusion_matrix(conf_mat, classes=['Class 0', 'Class 1','class 2'])

    # Compute the metrics
    dt_accuracy = EXACTITUDE(conf_mat)
    dt_specificity = SPECIFICITE(conf_mat)
    prec, dt_precision = PRECISION(conf_mat)
    rec, dt_recall = RAPPEL(conf_mat)
    dt_f1_score = F1_score(conf_mat)
    df_metrics = pd.DataFrame({'Accuracy': dt_accuracy, 'Specificity': dt_specificity, 'Precision': dt_precision, 'Recall': dt_recall, 'F1_score': dt_f1_score,'Execution time':Decision_Tree_time}, index=[0])

    return fig, conf_mat, df_metrics , decision_tree

#-----------------------------------------Random_forest-----------------------------------------#
def execute_Rf(n_trees, min_samples_split, max_depth, n_features=None):
    dataset = Preprocessing_1()
    train_set , test_set = custom_train_test_split(dataset.values, test_size=0.2, random_state=0)

    train_set = pd.DataFrame(train_set, columns=dataset.columns)
    test_set = pd.DataFrame(test_set, columns=dataset.columns)

    # Transforming the train and test sets into lists
    x_train = train_set.iloc[:, :-1]
    y_train = train_set.iloc[:, -1]
    x_test = test_set.iloc[:, :-1]
    y_test = test_set.iloc[:, -1]
    
    # Create an instance of the RandomForest class
    random_forest = RandomForest(n_trees, min_samples_split, max_depth, n_features=n_features)

    # Fit the model on the training set
    random_forest.fit(np.array(x_train), np.array(y_train))

    random_forest_start = time.time()
    # Predict on the test set
    y_pred_rf = random_forest.predict(np.array(x_test))
    random_forest_end = time.time()
    Random_Forest_time = random_forest_end - random_forest_start

    # Compute the confusion matrix
    conf_mat = confusion_matrix(y_test, y_pred_rf, len(np.unique(y_test)))
    fig = plot_confusion_matrix(conf_mat, classes=['Class 0', 'Class 1','class 2'])

    # Compute the metrics
    rf_accuracy = EXACTITUDE(conf_mat)
    rf_specificity = SPECIFICITE(conf_mat)
    prec, rf_precision = PRECISION(conf_mat)
    rec, rf_recall = RAPPEL(conf_mat)
    rf_f1_score = F1_score(conf_mat)

    df_metrics = pd.DataFrame({'Accuracy': rf_accuracy, 'Specificity': rf_specificity, 'Precision': rf_precision, 'Recall': rf_recall, 'F1_score': rf_f1_score,'Execution time':Random_Forest_time}, index=[0])

    return fig, conf_mat, df_metrics , random_forest
     
# -----------------------------------------Kmeans-----------------------------------------# 
def run_kmeans(k ,dataset ,demention):
    # dataset,_ = Preprocessing(0.7 , Norm=1)
    X = dataset.values
    kmeans = K_MEANS(k=k ,max_iter=100, metric=euclidean_distance)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    silhouette_score = kmeans.silhouette_score(X)
    davies_bouldin_score = kmeans.davies_bouldin_score(X)
    calinski_harabasz_score = kmeans.calinski_harabasz_score(X)
    # create a dataframe to display the metrics
    df_metrics = pd.DataFrame({'Silhouette Score': silhouette_score, 'Davies Bouldin Score': davies_bouldin_score, 'Calinski Harabasz Score': calinski_harabasz_score}, index=[0])
    plot_clusters = kmeans.plot_clusters(X,labels ,demention)
    return kmeans, df_metrics , plot_clusters

# -----------------------------------------DBSCAN-----------------------------------------#
def run_dbscan(eps, min_samples , dataset):
    # dataset,dataset_class = Preprocessing(0.7,)
    dbscan = DBSCAN_(eps=eps, min_samples=min_samples)
    X = dataset.values

    dbscan.fit(X)
    labels = dbscan.labels
    if len(set(dbscan.labels)) == 1:
        return 0, 0, 0
    else:
        silhouette_score = dbscan.silhouette_score(X)
 
        davies_bouldin_score = dbscan.davies_bouldin_score(X)
        calinski_harabasz_score = dbscan.calinski_harabasz_score(X)

        df_metrics = pd.DataFrame({'Silhouette Score': silhouette_score, 'Davies Bouldin Score': davies_bouldin_score,
                                    'Calinski Harabasz Score': calinski_harabasz_score}, index=[0])
        plot_clusters = dbscan.plot_clusters(X,labels ,demention=2)
        return dbscan, df_metrics , plot_clusters