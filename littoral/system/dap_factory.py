import numpy as np

##############################
# Clustering Model Factories #
##############################

class Factory:
  def __init__(self, n_clusters=2):
    self.n_clusters = n_clusters
    self.clf = None
  
  def fit(self, X):
    pass

  def cluster_points(self, X):
    clusters = [[] for _ in range(self.n_clusters)]
    for i, label in enumerate(self.labels):
        clusters[label].append(X[i])
    
    return clusters

class KMeansFactory(Factory):
  def __init__(self, n_clusters=2):
    from sklearn.cluster import KMeans
    super().__init__(n_clusters)
    self.clf = KMeans(n_clusters, n_init='auto', init='k-means++')
  
  def fit(self, X):
    self.clf.fit(X)
    self.labels = self.clf.labels_
    self.cluster_centers = self.clf.cluster_centers_
  
class KMedoidsFactory(Factory):
  def __init__(self, n_clusters=2):
    from sklearn_extra.cluster import KMedoids
    super().__init__(n_clusters)
    self.clf = KMedoids(n_clusters)
  
  def fit(self, X):
    self.clf.fit(X)
    self.labels = self.clf.labels_
    self.cluster_centers = self.clf.cluster_centers_

class CMeansSKFactory(Factory):
  def __init__(self, n_clusters=2):
    super().__init__(n_clusters)

  def fit(self, X):
    from skfuzzy.cluster import cmeans
    self.cluster_centers, u, _, _, _, _, _ = \
      cmeans(X.T, c=self.n_clusters, m=2, error=0.005, maxiter=1000)
    self.labels = np.argmax(u, axis=0)    
    
class CMeansFCFactory(Factory):
  def __init__(self, n_clusters=2):
    from littoral.cluster.fcluster import FCluster
    super().__init__(n_clusters)
    self.clf = FCluster(n_clusters, fuzzines=2, error=0.005, max_iter=1000)

  def fit(self, X):
    u, self.cluster_centers = self.clf.fit(X)
    self.labels = np.argmax(u, axis=-1)    
  
class GKFactory(Factory):
  def __init__(self, n_clusters=2):
    from littoral.cluster.fcluster import FCluster
    super().__init__(n_clusters)
    self.clf = \
      FCluster(n_clusters, fuzzines=2, error=0.005, max_iter=1000, method='Gustafson–Kessel')
  
  def fit(self, X):
    u, self.cluster_centers = self.clf.fit(X)
    self.labels = np.argmax(u, axis=-1)    

class RandFactory(Factory):
  def __init__(self, n_clusters=2):
    super().__init__(n_clusters)
  
  def fit(self, X):
    from sklearn.metrics import pairwise_distances
    from dap_utils import generate_ed_coords

    self.cluster_centers = generate_ed_coords(self.n_clusters)
    dists = pairwise_distances(X, self.cluster_centers, n_jobs=-1)
    self.labels = [np.argmin(dist) for dist in dists]

##############################