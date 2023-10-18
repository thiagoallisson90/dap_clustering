import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from fcluster import FCluster
from skfuzzy.cluster import cmeans
from dap_utils import generate_ed_coords

##################
# DAP Clustering #
##################

def clustering(X, n_clusters, labels):
    cluster_points = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        cluster_points[label].append(X[i])
    
    return cluster_points

def cm_cluster(X, n_clusters=2, m=2, error=0.005, max_iter=1000):
    cntr, u, _, _, _, _, _ = cmeans(X.T, c=n_clusters, m=m, error=error, maxiter=max_iter)
    labels = np.argmax(u, axis=0)    
    cluster_points = clustering(X, n_clusters, labels)

    return cntr, labels, cluster_points

def gk_cluster(X, n_clusters=2, m=2, error=1e-5, max_iter=150):
    clf = FCluster(n_clusters=n_clusters, method='Gustafson–Kessel', fuzzines=m, 
                   error=error, max_iter=max_iter)
    u, cntr = clf.fit(X)
    labels = np.argmax(u, axis=-1)
    cluster_points = clustering(X, n_clusters, labels)
    
    return cntr, labels, cluster_points

def kmeans_cluster(X, n_clusters=2):
    clf = KMeans(n_clusters, n_init='auto', init='k-means++').fit(X)
    cluster_points = clustering(X, n_clusters, clf.labels_)

    return clf.cluster_centers_, clf.labels_, cluster_points

def kmedoids_cluster(X, n_clusters=2):
    clf = KMedoids(n_clusters).fit(X)
    cluster_points = clustering(X, n_clusters, clf.labels_)

    return clf.cluster_centers_, clf.labels_, cluster_points

def rand_cluster(X, n_clusters=16):
    cntr = generate_ed_coords(n_clusters, seed=None)
    dists = pairwise_distances(X, cntr, n_jobs=-1)
    labels = [np.argmin(dist) for dist in dists]
    cluster_points = clustering(X, n_clusters, labels)
    
    return cntr, labels, cluster_points

run_model = {
    'kmeans': kmeans_cluster,
    'kmedoids':  kmedoids_cluster,
    'cmeans': cm_cluster,
    'gk': gk_cluster,
    'rand': rand_cluster,
}

##################