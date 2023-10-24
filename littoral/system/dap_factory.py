import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from skfuzzy.cluster import cmeans

from littoral.algorithms.dap_gk import GK

##############################
# Clustering Model Factories #
##############################

class Factory:
  def __init__(self, n_clusters):
    self.n_clusters = n_clusters
    self.cluster_centers = None
    self.labels = None
    self.clf = None

  def fit(self, X):
    pass

  def cluster_points(self, X):
    clusters = [[] for _ in range(self.n_clusters)]
    for i, label in enumerate(self.labels):
        clusters[label].append(X[i])
    
    return clusters

  def calc_wcss(self, X):
    clusters = self.cluster_points(X)
    self.wcss = 0

    for i, points in enumerate(clusters):
      cluster_distances = [euclidean(self.cluster_centers[i], point) for point in points]
      self.wcss = self.wcss + np.sum([dist ** 2 for dist in cluster_distances])
   
class KMeansFactory(Factory):
  def __init__(self, n_clusters=2):
    super().__init__(n_clusters)
    self.name = 'K-Means'
    self.clf = KMeans(n_clusters, n_init='auto', init='k-means++')

  def fit(self, X):
    self.clf.fit(X)
    self.labels = self.clf.labels_
    self.cluster_centers = self.clf.cluster_centers_
    self.calc_wcss(X)
    return self
  
class KMedoidsFactory(Factory):
  def __init__(self, n_clusters=2):
    super().__init__(n_clusters)
    self.name = 'K-Medoids'
    self.clf = KMedoids(n_clusters)
  
  def fit(self, X):
    self.clf.fit(X)
    self.labels = self.clf.labels_
    self.cluster_centers = self.clf.cluster_centers_
    self.calc_wcss(X)
    return self

class CMeansFactory(Factory):
  def __init__(self, n_clusters=2):
    super().__init__(n_clusters)
    self.name = 'Fuzzy C-Means'

  def fit(self, X):
    self.cluster_centers, u, _, _, _, _, _ = \
      cmeans(X.T, c=self.n_clusters, m=2, error=0.005, maxiter=1000)
    self.labels = np.argmax(u, axis=0)
    self.calc_wcss(X)
    return self
      
class GKFactory(Factory):
  def __init__(self, n_clusters=2):
    super().__init__(n_clusters)
    self.name = 'Gustafson-Kessel'
    self.clf = GK(n_clusters, m=2)

  def fit(self, X):
    self.clf.fit(X)
    self.labels = np.argmax(self.clf.u, axis=0)
    self.cluster_centers = self.clf.centers
    self.calc_wcss(X)
    return self

class RandFactory(Factory):
  def __init__(self, n_clusters=2):
    super().__init__(n_clusters)
    self.name = 'Rand' + str(n_clusters)
  
  def fit(self, X):
    from littoral.system.dap_utils import generate_ed_coords

    self.cluster_centers = generate_ed_coords(self.n_clusters)
    dists = pairwise_distances(X, self.cluster_centers, n_jobs=-1)
    self.labels = [np.argmin(dist) for dist in dists]
    
    self.wcss = 0
    for dist in dists:
      self.wcss = self.wcss + np.sum(np.min(dist) ** 2)

    return self

##############################