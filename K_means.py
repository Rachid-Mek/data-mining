from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA 
from Metrics_distance import *

# --------------------------------------------------------------------------------------------------------------
class K_means_: # K-means clustering algorithm
    def __init__(self, k=3, max_iter=100 , metric =euclidean_distance): # initialize the parameters
        self.k = k # number of clusters
        self.max_iter = max_iter # maximum number of iterations
        self.metric = metric # distance metric
# ------------------------------------------------------------------------------------------------------------------------
    def fit(self, X): # fit the data
        self.centroids = {} # initialize the centroids
        self.centroids = self.calculate_centroids(X) # calculate the centroids
        # Optimize centroids 
        for i in range(self.max_iter): # iterate over the maximum number of iterations
            self.clusters = {} # initialize the clusters
            for i in range(self.k): # initialize the clusters
                self.clusters[i] = [] 
            # Assign data points to the closest centroid
            for x in X:
                closest_centroid = self.find_closest_cluster(x)
                self.clusters[closest_centroid].append(x)
            # Calculate new centroids from the clusters
            for cluster in self.clusters:
                self.centroids[cluster] = np.average(self.clusters[cluster], axis=0)
        return self.clusters
# ------------------------------------------------------------------------------------------------------------------------
    def predict(self, X): # predict the cluster of each point
        distances = [np.linalg.norm(X - self.centroids[centroid]) for centroid in self.centroids] 
        closest_centroid = distances.index(min(distances)) 
        return closest_centroid
    def predict_all(self, X): # predict the cluster of each point
        predictions = []
        for x in X:
            distances = [np.linalg.norm(x - self.centroids[centroid]) for centroid in self.centroids] 
            closest_centroid = distances.index(min(distances)) 
            predictions.append(closest_centroid)
        return predictions
    def accuracy(self, y_test, y_pred):
        return np.sum(y_pred == y_test) / len(y_test)
    

# ------------------------------------------------------------------------------------------------------------------------
    def calculate_centroids(self, X): # initialize the centroids
        self.centroids = {} # initialize the centroids
        for i in range(self.k): # initialize the centroids
            self.centroids[i] = X[np.random.choice(range(len(X)))]
        return self.centroids
    
    def find_cluster(self, X): # find the cluster of each point
        self.clusters = {}
        for i in range(self.k):
            self.clusters[i] = []
        for x in X:
            # distances = [np.linalg.norm(x - self.centroids[centroid]) for centroid in self.centroids] 
            distances = []
            for centroid in self.centroids:
                distances.append(self.metric(x, self.centroids[centroid]))
            closest_centroid = distances.index(min(distances))
            self.clusters[closest_centroid].append(x)
        return self.clusters
    
    def find_closest_cluster(self, x): # find the closest cluster to a point
        distances = []
        for centroid in self.centroids:
            distances.append(np.linalg.norm(x - self.centroids[centroid]))
        closest_centroid = distances.index(min(distances))
        return closest_centroid
         
# ------------------------------------------------------------------------------------------------------------------------
    # silhouette score is a measure of how similar an object is to its own cluster (cohesion) 
    # compared to other clusters (separation)
    def silhouette_score(self, X):
        # Calculate the silhouette score for each sample
        labels = [self.predict(x) for x in X]
        # Calculate the silhouette score for the whole dataset
        return np.mean([self.silhouette_sample(X[i], labels[i]) for i in range(len(X))])

    def silhouette_sample(self, x, label):
        # Calculate the average distance within the same cluster (a)
        a = np.mean([np.linalg.norm(x - x_) for x_ in self.clusters[label]])

        # Calculate the average distance to the nearest cluster (b)
        n_clusters = len(self.clusters)
        b_values = [np.mean([np.linalg.norm(x - x_) for x_ in self.clusters[i]]) for i in range(n_clusters) if i != label]
        b = min(b_values) if b_values else 0

        # Calculate the silhouette score
        return (b - a) / max(a, b)
    def inter_cluster_distance(self, X):
        # Calculate the average distance between each cluster and all other clusters
        sigma_R = np.zeros(self.k)
        for i in range(self.k):
            cluster_points_i = np.array(self.clusters[i])
            cluster_center_i = cluster_points_i.mean(axis=0)
            sigma_R[i] = np.mean([np.linalg.norm(x - cluster_center_i) for x in self.clusters[i]])
        return np.mean(sigma_R)
    def intra_cluster_distance(self, X):
        # Calculate the average distance between each cluster and the next nearest cluster
        R = np.zeros(self.k)
        for i in range(self.k):
            cluster_points_i = np.array(self.clusters[i])
            cluster_center_i = cluster_points_i.mean(axis=0)
            R[i] = max([np.mean([np.linalg.norm(x - np.array(self.clusters[j]).mean(axis=0)) for x in self.clusters[i]]) for j in range(self.k) if j != i])
        return np.mean(R)
    def davies_bouldin_score(self, X):
        # from scratch
        # Calculate the average distance between each cluster and all other clusters
        sigma_R = np.zeros(self.k)
        for i in range(self.k):
            cluster_points_i = np.array(self.clusters[i])
            cluster_center_i = cluster_points_i.mean(axis=0)
            sigma_R[i] = np.mean([np.linalg.norm(x - cluster_center_i) for x in self.clusters[i]])
        # Calculate the average distance between each cluster and the next nearest cluster
        R = np.zeros(self.k)
        for i in range(self.k):
            cluster_points_i = np.array(self.clusters[i])
            cluster_center_i = cluster_points_i.mean(axis=0)
            R[i] = max([np.mean([np.linalg.norm(x - np.array(self.clusters[j]).mean(axis=0)) for x in self.clusters[i]]) for j in range(self.k) if j != i])
        # Calculate the Davies-Bouldin score
        return np.mean(sigma_R / R)
   
    def inertia(self, X):
            # Calculate the sum of squared distances of samples to their closest cluster center
            distances = np.linalg.norm(X - self.centroids[self.predict(X)], axis=1)
            return np.sum(distances ** 2)

    def calinski_harabasz_score(self, X):
        # Calculate the sum of squared distances of samples to their closest cluster center
        W = self.inertia(X)
        # Calculate the sum of squared distances of centroids to their closest cluster center
        centroids = np.array(list(self.centroids.values()))
        centroids_mean = centroids.mean(axis=0)
        centroids_mean = centroids_mean.reshape(1, -1)
        B = np.sum([np.linalg.norm(centroid - centroids_mean) for centroid in centroids])
        # Calculate the Calinski-Harabasz score
        return B / W * (len(X) - self.k) / (self.k - 1)
# ------------------------------------------------------------------------------------------------------------------------
    def plot_clusters(self, X):
        pca = PCA(n_components=2)
        pca.fit(X)
        labels =self.predict_all(X)

        X_pca = pca.transform(X)
        plt.figure(figsize=(6, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
        centers = self.centroids
        centers_array = np.array(list(centers.values()))
        centers_pca = pca.transform(centers_array)
        plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', s=200, alpha=0.5);
        
        # plot centers
        plt.title(f' k={len(self.clusters)} clusters , silhouette_score={self.silhouette_score(X)}')
        return plt