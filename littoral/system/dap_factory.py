import numpy as np
from littoral.system.dap_elbow import crisp_elbow, fuzzy_elbow
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from skfuzzy.cluster import cmeans
from littoral.algorithms.dap_fcluster import FCluster

##############################
# Clustering Model Factories #
##############################

class Factory:
  def __init__(self):
    self.n_clusters = 0
    self.clf = None
  
  def create(self, n_clusters=2):
    self.n_clusters = n_clusters
    return self

  def fit(self, X):
    pass

  def opt_k(self, X, ks=range(2, 30), method='elbow', metric='distortion'):
    pass

  def cluster_points(self, X):
    clusters = [[] for _ in range(self.n_clusters)]
    for i, label in enumerate(self.labels):
        clusters[label].append(X[i])
    
    return clusters

class KMeansFactory(Factory):
  def __init__(self):
    super().__init__()
    self.name = 'K-Means'
    self.optimization_methods = {
      'elbow': crisp_elbow,
    }
  
  def create(self, n_clusters=2):
    self.n_clusters=n_clusters
    self.clf = KMeans(n_clusters, n_init='auto', init='k-means++')
    return self

  def fit(self, X):
    self.clf.fit(X)
    self.labels = self.clf.labels_
    self.cluster_centers = self.clf.cluster_centers_
  
  def opt_k(self, X, ks=range(2, 30), method='elbow', metric='distortion'):
    clf = KMeans(n_init='auto', init='k-means++')
    return self.optimization_methods[method](X, ks, clf)    
  
class KMedoidsFactory(Factory):
  def __init__(self):
    super().__init__()
    self.name = 'K-Medoids'
    self.optimization_methods = {
      'elbow': crisp_elbow,
    }
  
  def create(self, n_clusters=2):
    self.clf = KMedoids(n_clusters, init='k-medoids++')
    return self
  
  def fit(self, X):
    self.clf.fit(X)
    self.labels = self.clf.labels_
    self.cluster_centers = self.clf.cluster_centers_
  
  def opt_k(self, X, ks=range(2, 30), method='elbow', metric='distortion'):
    clf = KMedoids(init='k-medoids++')
    return self.optimization_methods[method](X, ks, clf)

class CMeansSKFactory(Factory):
  def __init__(self):
    super().__init__()
    self.name = 'Fuzzy C-Means'
    self.optimization_methods = {
      'elbow': fuzzy_elbow,
    }

  def create(self, n_clusters=2):
    super().create(n_clusters)
    return self

  def fit(self, X):
    self.cluster_centers, u, _, _, _, _, _ = \
      cmeans(X.T, c=self.n_clusters, m=2, error=0.005, maxiter=1000)
    self.labels = np.argmax(u, axis=0)   
  
  def opt_k(self, X, ks=range(2, 30), method='elbow', metric='distortion'):
    pass
    
class CMeansFCFactory(Factory):
  def __init__(self):
    super().__init__()
    self.name = 'Fuzzy C-Means'
    self.optimization_methods = {
      'elbow': fuzzy_elbow,
    }

  def create(self, n_clusters=2):
    super().create(n_clusters)
    self.clf = FCluster(n_clusters, fuzzines=2, error=0.005, max_iter=1000)
    return self

  def fit(self, X):
    u, self.cluster_centers = self.clf.fit(X)
    self.labels = np.argmax(u, axis=-1)    
  
  def opt_k(self, X, ks=range(2, 30), method='elbow', metric='distortion'):
    pass
  
class GKFactory(Factory):
  def __init__(self):
    super().__init__()
    self.name = 'Gustafson-Kessel'
    self.optimization_methods = {
      'elbow': fuzzy_elbow,
    }
  
  def create(self, n_clusters=2):
    super().create(n_clusters)
    self.clf = FCluster(n_clusters, fuzzines=2, error=0.005, max_iter=1000, method='Gustafson–Kessel')
    return self

  def fit(self, X):
    u, self.cluster_centers = self.clf.fit(X)
    self.labels = np.argmax(u, axis=-1)    
  
  def opt_k(self, X, ks=range(2, 30), method='elbow', metric='distortion'):
    pass

class RandFactory(Factory):
  def __init__(self):
    super().__init__()
    self.name = 'Rand'
  
  def create(self, n_clusters=2):
    from littoral.system.dap_utils import generate_ed_coords
    super().create(n_clusters)
    self.name = self.name + str(n_clusters)
    self.cluster_centers = generate_ed_coords(self.n_clusters)
    return self
  
  def fit(self, X):
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(X, self.cluster_centers, n_jobs=-1)
    self.labels = [np.argmin(dist) for dist in dists]

##############################