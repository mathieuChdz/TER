from collections import Counter

import pandas as pd
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report

import seaborn as sns
from matplotlib import pyplot as plt

import cupy as cp

import time

# Model

class KmeansCPU():
    """
    Implements the K-means algorithm.
    """
    
    def __init__(self, k=3):
        """
        Constructor.
        
        Parameters
        ----------
        
        k : int
            Number of clusters to be constructed.
        """
        
        self.k = k
        self.clusters = None
        self.centroids = None
        
    
    def _initialize_centroids(self, X):
        """
        Initialize k centroids from a normal distribution.
        The mean of the distribution is the mean of the data X.
        The std of the distribution is 1.0.
        The k centroids are stored into an array.
        
        Parameters
        ----------
        X : ndarray
            Initial data of dim N x p.
            
        Returns
        -------
        clusters : ndarray
            Array of clusters assigned to the points of X 
            (dim X.shape[0]).
        """
        
        centre = np.mean(X, axis=0)
        centroids = np.random.normal(loc=centre,
                                     scale=1,
                                     size=(self.k, X.shape[1]))
        
        return centroids

    
    def _assign_clusters(self, X, centroids):
        """
        Given the data X and the current centroids,
        this function computes the cluster assignment
        of the data.
        
        Each data point of X is assigned to the cluster
        whose centroid is the closest to it.
        
        Parameters
        ----------
        X : ndarray
            Initial data of dim N x p.
        centroids : ndarray
            Array of k centroids (dim k x X.shape[1]).
            
        Returns
        -------
        clusters : ndarray
            Array of clusters assigned to the points of X 
            (dim X.shape[0]).
        """
        
        distances = np.zeros(shape=(X.shape[0], self.k))

        for i in range(self.k):
            
            # distance to the i-th centroid
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)

        clusters = distances.argmin(axis=1)
        
        return clusters

    
    def _compute_centroids(self, X, clusters):
        """
        Compute the new centroids of the data X according to
        their cluster assignment clusters.
        
        Parameters
        ----------
        X : ndarray
            Initial data of dim N x p
        clusters : ndarray
            Array of clusters assigned to the points of X 
            (dim X.shape[0]).
        
        Returns
        -------
        centroids : ndarray
            Array of k centroids (dim k x X.shape[1]).
        """
        
        centroids = -np.ones(shape=(self.k, X.shape[1]))

        for i in range(self.k):

            X_class = X[clusters==i]
            centroids[i] = X_class.mean(axis=0)
        
        return centroids
    
    
    def fit(self, X, stop_dist=0.01, max_iter=1000):
        """
        Implements the K-means algorithm for the data X.
        
        First, the algorithm initializes random centroids
        via the _initialize_centroids methods. 
        Then, the algorithm alternates between the
        _assign_clusters and the _compute_centroids methods
        until a stop criterion is met.
        The algorithm stops when the distances between the 
        previous and new centroids hasn't changed too much,
        i.e., is less than stop_dist, or when the maximum
        number of iterations has been reached.
        The method re-assigns the attributes self.clusters
        and self.centroids to the clusters and centroids
        that it has computed.
        
        Parameters
        ----------
        X : ndarray
            Initial data of dim N x p.
        stop_dist : float
            Minimal distance between previous and new centroid
        max_iter : int
            maximum number of iterations.
        
        Returns
        -------
        clusters, centroids : ndarray, ndarray
            Array of clusters assigned to the points of X 
            (dim X.shape[0]).
            Array of k centroids (dim k x X.shape[1]).
        """
        it = 0
        dist = np.inf * np.ones(shape=(self.k))
        stop_dist = stop_dist * np.ones(shape=(self.k))
        centroids = self._initialize_centroids(X)
        
        # print("Fitting in progress:")
        
        while (dist > stop_dist).any() and it <= max_iter:
            
            # print(".", end="")
            clusters = self._assign_clusters(X, centroids)
            new_centroids = self._compute_centroids(X, clusters)
            dist = np.linalg.norm(centroids - new_centroids, axis=1)
            centroids = new_centroids
            it += 1
        
        # print(" total itérations : ", it)

        self.clusters = clusters
        self.centroids = centroids
        
        
    def predict(self, x_new):
        """
        Computes the cluster prediction c_new_hat associated 
        to a new point x_new. The cluster c_new_hat is the one 
        whose centroid is the closest to x_new.

        Parameters
        ----------
        x_new : Union[ndarray, list]
            New point of dim p to be classified by the $K$-means algo.
            
        Returns
        -------
        c_new_hat : int
            Cluster prediction for the point x_new.
        """
        
        dist = np.zeros(shape=(self.k))

        for i in range(self.k):
            
            # distance to the i-th centroid
            dist[i] = np.linalg.norm(x_new - self.centroids[i])

        c_new_hat = dist.argmin()
        
        return c_new_hat

# EXECUTION EN GPU :

class KmeansGPU():
    def __init__(self, k=3):
        self.k = k
        self.clusters = None
        self.centroids = None

    def _initialize_centroids(self, X):
        centre = cp.mean(X, axis=0)
        centroids = cp.random.normal(loc=centre, scale=1, size=(self.k, X.shape[1]))
        return centroids

    def _assign_clusters(self, X, centroids):
        distances = cp.zeros(shape=(X.shape[0], self.k))
        for i in range(self.k):
            distances[:, i] = cp.linalg.norm(X - centroids[i], axis=1)
        clusters = distances.argmin(axis=1)
        return clusters

    def _compute_centroids(self, X, clusters):
        centroids = -cp.ones(shape=(self.k, X.shape[1]))
        for i in range(self.k):
            X_class = X[clusters == i]
            centroids[i] = X_class.mean(axis=0)
        return centroids

    def fit(self, X, stop_dist=0.01, max_iter=1000):
        it = 0
        dist = cp.inf * cp.ones(shape=(self.k))
        stop_dist = stop_dist * cp.ones(shape=(self.k))
        centroids = self._initialize_centroids(X)
        # print("Fitting in progress:")
        while (dist > stop_dist).any() and it <= max_iter:
            # print(".", end="")
            clusters = self._assign_clusters(X, centroids)
            new_centroids = self._compute_centroids(X, clusters)
            dist = cp.linalg.norm(centroids - new_centroids, axis=1)
            centroids = new_centroids
            it += 1
        self.clusters = clusters
        self.centroids = centroids

    def predict(self, x_new):
        x_new = cp.asarray(x_new)
        dist = cp.zeros(shape=(self.k))
        for i in range(self.k):
            dist[i] = cp.linalg.norm(x_new - self.centroids[i])
        c_new_hat = dist.argmin()
        return c_new_hat



# ## Results
# ### Execution in CPU
def result_execution(X, y, k=3, bool_gpu=False):
    """
    Execute the K-means algorithm on the data X."
    """
    if bool_gpu:
        X = cp.asarray(X)
        y = cp.asarray(y)
        kmeans = KmeansGPU(k=k)
    else:
        kmeans = KmeansCPU(k=k)
    centroids = kmeans._initialize_centroids(X)
    kmeans.fit(X, max_iter=100)
    clusters, centroids = kmeans.clusters, kmeans.centroids
    kmeans.predict([-6.59672862, -6.42369954])
    
    # plot_clusters(X, y, clusters, centroids)
    # plot_original_cluster()

    if bool_gpu:
        y_new = -cp.ones(X.shape[0])
    else:
        y_new = -np.ones(X.shape[0])

    y_new[y==0] = 2
    y_new[y==1] = 1
    y_new[y==2] = 0

    # print(classification_report(y_new, clusters))

def plot_clusters(X, y, clusters, centroids):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[clusters==0][:, 0], X[clusters==0][:, 1],
                color="C0", alpha=1, marker = 'x', label="class 0")
    plt.scatter(centroids[0][0], centroids[0][0],
                color="black", alpha=0.5, s=100)
    plt.scatter(X[clusters==1][:, 0], X[clusters==1][:, 1],
                color="C1", alpha=1, marker = 'x', label="class 1")
    plt.scatter(centroids[1][0], centroids[1][1],
                color="black", alpha=0.5, s=100)
    plt.scatter(X[clusters==2][:, 0], X[clusters==2][:, 1],
                color="C2", alpha=1, marker = 'x', label="class 2")
    plt.scatter(centroids[2][0], centroids[2][1],
                color="black", alpha=0.5, s=100, label="centroids")
    plt.xlabel("X_1", fontsize=14)
    plt.ylabel("X_2", fontsize=14)
    plt.legend(fontsize=14)
    plt.title("K-Means", fontsize=16)
    plt.savefig("Graphs/kmeans.jpg")
    plt.show()

def plot_original_cluster():
    # original clusters
    plt.figure(figsize=(10, 6))

    plt.scatter(X[y==0][:, 0], X[y==0][:, 1], 
                color="C0", alpha=1, marker = 'x', label="class 0")

    plt.scatter(X[y==1][:, 0], X[y==1][:, 1], 
                color="C1", alpha=1, marker = 'x', label="class 1")

    plt.scatter(X[y==2][:, 0], X[y==2][:, 1], 
                color="C2", alpha=1, marker = 'x', label="class 2")


    plt.xlabel("X_1", fontsize=14)
    plt.ylabel("X_2", fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Original clusters", fontsize=16)

    # plt.savefig("Graphs/kmeans.jpg")
    plt.show()


if __name__ == "__main__":
    sns.set_theme()

    NB_ITERATION = 10
    point_range = range(500000, 5000001, 500000)

    # Results for CPU
    result_global_mean_dict_cpu = {}
    for i in point_range:
        result_list_i = []
        for _ in range(NB_ITERATION):
            # Generate data
            X, y = make_blobs(n_samples=i, n_features=2, centers=3, cluster_std=4.0, random_state=42)
            start = time.time()
            result_execution(X, y, k=3, bool_gpu=False)
            end = time.time()
            result_list_i.append(end - start)
        result_global_mean_dict_cpu[i] = np.mean(result_list_i)
        print("CPU mean time for ", i, " points: ", result_global_mean_dict_cpu[i])

    # Results for GPU
    result_global_mean_dict_gpu = {}
    for i in point_range:
        result_list_i = []
        for _ in range(NB_ITERATION):
            # Generate data
            X, y = make_blobs(n_samples=i, n_features=2, centers=3, cluster_std=4.0, random_state=42)
            start = time.time()
            result_execution(X, y, k=3, bool_gpu=True)
            end = time.time()
            result_list_i.append(end - start)
        result_global_mean_dict_gpu[i] = np.mean(result_list_i)
        print("GPU mean time for ", i, " points: ", result_global_mean_dict_gpu[i])

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(list(result_global_mean_dict_cpu.keys()), list(result_global_mean_dict_cpu.values()), label="CPU", marker='o')
    plt.plot(list(result_global_mean_dict_gpu.keys()), list(result_global_mean_dict_gpu.values()), label="GPU", marker='.')
    plt.xlabel("Number of points", fontsize=14)
    plt.ylabel(f"Execution time (s) (mean of {NB_ITERATION} iterations)", fontsize=14)
    plt.legend(fontsize=14)
    plt.title("Execution time of K-means algorithm (CPU vs GPU)", fontsize=16)
    plt.savefig("Graphs/kmeans_cpu_vs_gpu_execution_times.jpg")
    plt.show()