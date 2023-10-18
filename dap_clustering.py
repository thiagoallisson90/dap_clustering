##################
# DAP Clustering #
##################

"""
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
"""

def kmeans_cluster(X, n_clusters=2):
    from dap_factory import KMeansFactory

    clf = KMeansFactory(n_clusters)
    clf.fit(X)

    return clf.cluster_centers, clf.labels, clf.cluster_points(X)

def kmedoids_cluster(X, n_clusters=2):
    from dap_factory import KMedoidsFactory

    clf = KMedoidsFactory(n_clusters)
    clf.fit(X)

    return clf.cluster_centers, clf.labels, clf.cluster_points(X)

def cm_cluster(X, n_clusters=2):
    from dap_factory import CMeansSKFactory

    clf = CMeansSKFactory(n_clusters)
    clf.fit(X)

    return clf.cluster_centers, clf.labels, clf.cluster_points(X)

def gk_cluster(X, n_clusters=2):
    from dap_factory import GKFactory

    clf = GKFactory(n_clusters)
    clf.fit(X)

    return clf.cluster_centers, clf.labels, clf.cluster_points(X)

def rand_cluster(X, n_clusters=2):
    from dap_factory import RandFactory

    clf = RandFactory(n_clusters)
    clf.fit(X)

    return clf.cluster_centers, clf.labels, clf.cluster_points(X)

run_clustering = {
    'kmeans': kmeans_cluster,
    'kmedoids':  kmedoids_cluster,
    'cmeans': cm_cluster,
    'gk': gk_cluster,
    'rand': rand_cluster,
}

##################