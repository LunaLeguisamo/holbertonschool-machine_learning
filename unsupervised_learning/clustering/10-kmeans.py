#!/usr/bin/env python3
"""
Function that performs K-means on a dataset
"""

from sklearn.cluster import KMeans


def kmeans(X, k):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
    k is the number of clusters
    The only import you are allowed to use is import sklearn.cluster
    Returns: C, clss
    C is a numpy.ndarray of shape (k, d) containing the centroid means
    for each cluster
    clss is a numpy.ndarray of shape (n,) containing the index of
    the cluster in C that each data point belongs to
    """
    kmeans_model = KMeans(n_clusters=k, random_state=0)
    kmeans_model.fit(X)
    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_
    return C, clss
