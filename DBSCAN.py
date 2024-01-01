from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
from Metrics_distance import *
# --------------------------------------------------------------------------------------------------------------

class DBSCAN_:
    def __init__(self, eps, min_samples,metric=euclidean_distance):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        self.metric = metric

    def fit(self, X):
        self.labels = np.full(X.shape[0], -1, dtype=int)
        cluster_label = 0

        for i in range(X.shape[0]):
            if self.labels[i] != -1:
                continue  # Skip points already visited

            neighbors = self.get_neighbors(X, i)

            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # Mark as noise
            else:
                self.expand_cluster(X, i, neighbors, cluster_label)
                cluster_label += 1

        return self

    # def get_neighbors(self, X, idx):
    #     distances = np.linalg.norm(X - X[idx], axis=1)
    #     return np.where(distances <= self.eps)[0]
    def get_neighbors(self, X, index):
        neighbors = []
        for i in range(X.shape[0]):
            if self.metric(X[index], X[i]) <= self.eps:
                neighbors.append(i)
        return np.array(neighbors)
    
    def clusters(self):
        # create cluster with points from X
        clusters = []
        for i in range(len(self.labels)):
            if self.labels[i] == -1:
                continue
            if self.labels[i] not in clusters:
                clusters.append(self.labels[i])
        return clusters

    def expand_cluster(self, X, seed, neighbors, cluster_label):
        self.labels[seed] = cluster_label

        i = 0
        while i < len(neighbors):
            current_point = neighbors[i]

            if self.labels[current_point] == -1: # unvisited
                self.labels[current_point] = cluster_label
                new_neighbors = self.get_neighbors(X, current_point)

                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            elif self.labels[current_point] == 0:
                self.labels[current_point] = cluster_label

            i += 1

 
     
#  ------------------------------ EVALUATION METRICS ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
    def silhouette_score(self, X):
        # labels = np.concatenate([np.full(len(cluster), i) for i, cluster in enumerate(self.clusters)])
        labels = self.labels
        silhouette_scores = []
        
        for i, x in enumerate(X):
            a_i = np.mean([np.linalg.norm(x - x_j) for x_j in X[labels == labels[i]]])
            b_i = min([np.mean([np.linalg.norm(x - x_k) for x_k in X[labels == label]]) for label in set(labels) - {labels[i]}])
            silhouette_scores.append((b_i - a_i) / max(a_i, b_i))

        return np.mean(silhouette_scores)
    def calculate_average_distance(self, x, points):
        if len(points) == 0:
            return 0
        return np.mean([self.metric(x, p) for p in points])

    # ------------------------------------------------------------------------------------------------------------------------
    def davies_bouldin_score(self, X):
        labels = self.labels
        n_clusters = len(set(labels) - {-1})  # Exclude noise points
        cluster_centers = [np.mean(X[labels == i], axis=0) for i in range(n_clusters)]

        sigma_R = []
        for i in range(n_clusters):
            cluster_i_points = X[labels == i]
            if len(cluster_i_points) == 0:
                continue  # Skip clusters with no points
            distances_i = np.linalg.norm(cluster_i_points - cluster_centers[i], axis=1)
            sigma_R_i = np.mean([np.linalg.norm(cluster_i_points[j] - cluster_centers[i]) for j in range(len(cluster_i_points))])
            sigma_R.append(sigma_R_i)

        R = []
        for i in range(n_clusters):
            if len(sigma_R) > 1:  # Check if sigma_R has more than one element
                R_i = max([(sigma_R[i] + sigma_R[j]) / np.linalg.norm(cluster_centers[i] - cluster_centers[j]) for j in range(n_clusters) if i != j])
                R.append(R_i)

        return np.mean(sigma_R / R)  


    def calinski_harabasz_score(self, X):
        labels = self.labels
        n_clusters = len(set(labels) - {-1})  # Exclude noise points
        cluster_centers = [np.mean(X[labels == i], axis=0) for i in range(n_clusters)]

        sigma_T = np.mean([np.mean([np.linalg.norm(X[i] - cluster_centers[j]) for i in range(len(X))]) for j in range(n_clusters)])

        sigma_B = np.mean([len(X[labels == i]) * np.linalg.norm(cluster_centers[i] - np.mean(X, axis=0)) for i in range(n_clusters)])

        return (sigma_B / sigma_T) * (len(X) - n_clusters) / (n_clusters - 1)

    
# ------------------------------------------------------------------------------------------------------------------------ 
    def plot_clusters(self ,X):
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(X)
        
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.labels, cmap='viridis', marker='o', s=20)
        plt.title(f'DBSCAN Clusters (eps={self.eps}, min_samples={self.min_samples})')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.tight_layout()

        return plt 
         
 