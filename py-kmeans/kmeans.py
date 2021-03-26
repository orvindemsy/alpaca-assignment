import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class KMeans():
    '''
    Author: Orvin Demsy
    Date  : 2021-03-26
    
    Implementation of K-Means Algorithm for AlpacaJapan code challenge. K-Means is unsupervised which goal
    is to find n cluster from given data. The cluster is found within some iterations where each iteration distance 
    from centroids to each sample is computed. n cluster is hyperparameter and defined by the user.
    
    PARAMETERS:
    n_cluster : int, number of cluster to be created
    n_iter    : int, number of iteration to find cluster
    
    ATTRIBUTES:
    fitted    : boolean, flag to check whether data is fitted
    labels    : array, predicted cluster labels once data is fitted
    centroids : array, shape (n_cluster x 2) new centroids once data is fitted 
    
    METHODS:
    fit(X)    : fit X into model, return: centroids after iteration
    predict(X): using the new centroids, find predicted label, return: predicted labels
    plot(X)   : plot clustered X, once fitted and predicted labels is found 
    '''
    def __init__(self, n_clusters = 2, n_iter=20):
        if n_clusters < 1:
            raise AssertionError('n_clusters must be > 0')
            
        self.n_clusters  = n_clusters
        self.n_iter      = 1 if not n_iter else n_iter 
        self.fitted      = False
        self.labels      = None
        self.centroids   = None
        
    # === Distance each samples to centroids === #
    def _distance(self, X, centroids):
        X = np.array(X)
        dist = np.array([])
        
        for c in centroids:
            # using euclidean/norm 2 as distance metric
            temp = np.linalg.norm((X - c), 2, 1)[:, None]
            dist = np.hstack([dist, temp]) if len(dist) else temp
        
        # predicting labels
        pred_labels = np.argmin(dist, axis=1)
        
        return pred_labels
    
    # === Compute new centroids === #
    def _compute_centroids(self, X, labels):
        new_cent = np.array([])
        
        for i in range(self.n_clusters):
            temp_cent = X[labels == i].mean(axis=0)
            new_cent  = np.vstack([new_cent, temp_cent]) if len(new_cent) else temp_cent        
        
        return np.array(new_cent)
    
    def fit(self, X):
        if np.array(X).ndim != 2:
            raise AssertionError(f'expect 2D array got {np.array(X).ndim}D instead')
            
        # initialize centroids
        centroids = np.array(X)[np.random.choice(range(len(X)), self.n_clusters)]
        
        for n in range(self.n_iter):
            pred_labels = self._distance(X, centroids)
            centroids   = self._compute_centroids(X, pred_labels)
        
        self.labels    = pred_labels
        self.centroids = centroids
        self.fitted    = True
        
        return self.centroids
        
    def predict(self, X):
        if not self.fitted:
            raise AssertionError('centroids not found KMeans is not fitted yet')
        if np.array(X).ndim != 2:
            raise AssertionError(f'expect 2D array got {np.array(X).ndim}D instead')
        
        return self._distance(X, self.centroids)
    
    def plot(self, X):
        if not self.fitted:
            raise AssertionError('centroids not found KMeans is not fitted yet')
        if np.array(X).ndim != 2:
            raise AssertionError(f'expect 2D array got {np.array(X).ndim}D instead')
        
        fig, ax = plt.subplots()
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=self.labels, palette='deep', ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [f'cluster-{i}' for i in labels])
        plt.show()