##################
# DAP Clustering #
##################

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