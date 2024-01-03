from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances 
from Metrics_distance import *

# --------------------------------------------------------------------------------------------------------------
class K_MEANS:
    def __init__(self, k=3, max_iter=100, metric=combined_metric, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.metric = metric
        self.random_state = random_state
        

    def initialize_centroids(self, X):
        centroids = [X[np.random.choice(len(X))]]
        for _ in range(1, self.k):
            distances = np.array([min(self.metric(c, x) for c in centroids) for x in X])
            probabilities = distances / distances.sum()
            centroids.append(X[np.random.choice(len(X), p=probabilities)])
        return np.array(centroids)
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iter):
            self.clusters = [[] for _ in range(self.k)]
            for x in X:
                distances = [self.metric(x, c) for c in self.centroids]
                self.clusters[np.argmin(distances)].append(x)
            for i in range(self.k):
                if len(self.clusters[i]) != 0:
                    self.centroids[i] = np.mean(self.clusters[i], axis=0)
                else:
                    # Reinitialize centroid for empty cluster
                    self.centroids[i] = X[np.random.choice(len(X))]
 
        return self 
  
 
    
    def predict(self, X):
        return np.array([np.argmin([self.metric(x, c) for c in self.centroids]) for x in X])

    def inertia(self, X):
        return sum(np.min([self.metric(x, c) for c in self.centroids]) for x in X)
         
# ------------------------------------------------------------------------------------------------------------------------
    def silhouette_score(self, X):
        labels = self.predict(X)
        return np.mean([self.silhouette_sample(X[i], labels[i]) for i in range(len(X))])

    def silhouette_sample(self, x, label):
        cluster_dict = {i: np.array(self.clusters[i]) for i in range(len(self.clusters))}
        a = np.mean(np.linalg.norm(x - cluster_dict[label], axis=1))
        b_values = [np.mean(np.linalg.norm(x - cluster_dict[i], axis=1)) for i in range(len(self.clusters)) if
                    i != label]
        b = min(b_values) if b_values else 0
        return (b - a) / max(a, b)

    
    def davies_bouldin_score(self, X):
        cluster_centers = np.array([np.mean(cluster, axis=0) for cluster in self.clusters])

        # Compute pairwise distances between cluster centers
        cluster_distances = pairwise_distances(cluster_centers)

        scores = []

        for i in range(self.k):
            cluster_points_i = np.array(self.clusters[i])
            cluster_center_i = cluster_points_i.mean(axis=0)

            intra_cluster_distances = [np.linalg.norm(x - cluster_center_i) for x in self.clusters[i]]
            within_cluster_distance = np.mean(intra_cluster_distances)

            # Calculate the similarity index for the current cluster
            similarity_indices = []

            for j in range(self.k):
                if j != i:
                    inter_cluster_distance = np.max(cluster_distances[i, j])
                    similarity_indices.append((within_cluster_distance + np.mean([np.linalg.norm(x - cluster_centers[j]) for x in self.clusters[j]])) / inter_cluster_distance)

            # Append the average similarity index for the current cluster
            scores.append(np.mean(similarity_indices))

        # Return the average similarity index over all clusters
        return np.mean(scores)

 
    
    def calinski_harabasz_score(self, X):
        # Calculate the total number of data points
        total_samples = len(X)

        # Calculate the overall mean
        overall_mean = np.mean(X, axis=0)

        # Calculate the between-cluster variance
        between_cluster_variance = 0.0

        for i in range(self.k):
            cluster_size = len(self.clusters[i])
            cluster_center = np.mean(self.clusters[i], axis=0)
            between_cluster_variance += cluster_size * np.linalg.norm(cluster_center - overall_mean)**2

        between_cluster_variance /= (self.k - 1)

        # Calculate the within-cluster variance
        within_cluster_variance = 0.0

        for i in range(self.k):
            cluster_center = np.mean(self.clusters[i], axis=0)
            within_cluster_variance += np.sum([np.linalg.norm(x - cluster_center)**2 for x in self.clusters[i]])

        within_cluster_variance /= (total_samples - self.k)

        # Calculate the Calinski-Harabasz Index
        calinski_harabasz_index = between_cluster_variance / within_cluster_variance

        return calinski_harabasz_index
    # --------------------------------------------------------------------------------------------------------
    def plot_clusters(self, X ,labels, demention=2):
        if demention == 2:
            pca = PCA(n_components=2)
            pca.fit(X)
            X_pca = pca.transform(X)

            plt.figure(figsize=(6, 6))
            plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
            centroids_pca = pca.transform(self.centroids)
            plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red', label='Centroids')
            plt.xlabel('First Principal Component')#
            plt.ylabel('Second Principal Component')
            return plt
        elif demention == 3:
            pca_k3 = PCA(n_components=3)
            pca_k3.fit(X)
            X_pca_k3 = pca_k3.transform(X)

            # Plot 3D clusters for k=3
            plt.figure(figsize=(12, 6))

            # 3D plot
            ax = plt.subplot(1, 2, 2, projection='3d')
            ax.scatter(X_pca_k3[:, 0], X_pca_k3[:, 1], X_pca_k3[:, 2], c=labels, cmap='viridis')
            pca = PCA(n_components=3)
            centroids_pca = pca.fit_transform(self.centroids)

            ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], centroids_pca[:, 2], marker='X', s=200, c='red', label='Centroids')
            ax.set_xlabel('First Principal Component')
            ax.set_ylabel('Second Principal Component')
            ax.set_zlabel('Third Principal Component')
            ax.set_title('Clusters for k=3')
            return plt