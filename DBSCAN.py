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
    
    def calinski_harabasz_score(self, X):
        labels = self.labels
        centroids = np.array([np.mean(X[labels == label], axis=0) for label in set(labels)])
        n = len(X)
        k = len(centroids)

        numerator = sum([len(X[labels == label]) * self.metric(centroids[label], np.mean(X, axis=0)) for label in set(labels)])
        denominator = (k - 1) * sum([sum([self.metric(x, centroids[label]) ** 2 for x in X[labels == label]]) for label in set(labels)])

        return numerator / denominator if denominator != 0 else 0.0
    def davies_bouldin_score(self, X):
        labels = self.labels
        centroids = np.array([np.mean(X[labels == label], axis=0) for label in set(labels)])
        n = len(X)
        k = len(centroids)
        return sum([max([(self.calculate_average_distance(centroids[i], X[labels == i]) + self.calculate_average_distance(centroids[j], X[labels == j])) / self.metric(centroids[i], centroids[j]) for j in range(k) if j != i]) for i in range(k)]) / k