#!/usr/bin/env python3
"""
Function that performs agglomerative clustering on a dataset
"""

import scipy.cluster.hierarchy as sc
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
    dist is the maximum cophenetic distance for all clusters
    Performs agglomerative clustering with Ward linkage
    Displays the dendrogram with each cluster displayed in a
    different color
    Returns: clss, a numpy.ndarray of shape (n,) containing
    the cluster indices for each data point
    """
    link = sc.cluster.hierarchy.linkage(X, method='ward')
    clss = sc.cluster.hierarchy.fcluster(link, t=dist, criterion='distance')
    sc.cluster.hierarchy.dendrogram(link, color_threshold=dist)
    plt.show()
    return clss
