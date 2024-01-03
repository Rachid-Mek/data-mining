from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.metrics import pairwise_distances
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
    def plot_clusters(self, X ,labels, demention=2):
        if demention == 2:
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="plasma")
            # add centroids
            plt.xlabel("Feature 0")
            plt.ylabel("Feature 1")
            plt.title(f"Cluster Assignments using DBSCAN")
            return plt
        elif demention == 3:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap="plasma")
            ax.set_xlabel('Feature 0')
            ax.set_ylabel('Feature 1')
            ax.set_zlabel('Feature 2')
            plt.title(f"Cluster Assignments using DBSCAN")
            return plt
      
 
     
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
        cluster_centers = np.array([np.mean(X[self.labels == i], axis=0) for i in self.clusters()])

        # Compute pairwise distances between cluster centers
        cluster_distances = pairwise_distances(cluster_centers)

        scores = []

        for i in self.clusters():
            cluster_points_i = X[self.labels == i]
            cluster_center_i = cluster_points_i.mean(axis=0)

            intra_cluster_distances = [np.linalg.norm(x - cluster_center_i) for x in cluster_points_i]
            within_cluster_distance = np.mean(intra_cluster_distances)

            # Calculate the similarity index for the current cluster
            similarity_indices = []

            for j in self.clusters():
                if j != i:
                    inter_cluster_distance = np.max(cluster_distances[i, j])
                    similarity_indices.append((within_cluster_distance + np.mean([np.linalg.norm(x - cluster_centers[j]) for x in X[self.labels == j]])) / inter_cluster_distance)

            # Append the average similarity index for the current cluster
            scores.append(np.mean(similarity_indices))

        # Return the average similarity index over all clusters
        return np.mean(scores)

    
    def calinski_harabasz_score(self, X):
        n_clusters = len(self.clusters())  # Exclude noise points
        cluster_centers = [np.mean(X[self.labels == i], axis=0) for i in self.clusters()]

        overall_mean = np.mean(X, axis=0)

        # Calculate the total scatter matrix (sigma_T)
        sigma_T = np.sum([np.sum((X[i] - overall_mean) ** 2) for i in range(len(X))])

        # Calculate the between-cluster scatter matrix (sigma_B)
        sigma_B = np.sum([len(X[self.labels == i]) * np.sum((cluster_centers[i] - overall_mean) ** 2) for i in self.clusters()])

        # Calculate the Calinski-Harabasz Index
        calinski_harabasz_index = (sigma_B / sigma_T) * (len(X) - n_clusters) / (n_clusters - 1)

        return calinski_harabasz_index