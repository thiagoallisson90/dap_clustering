"""import numpy as np
from copy import deepcopy
from scipy.linalg import norm
from scipy.spatial.distance import cdist

class FCluster:
    def __init__(self, n_clusters=2, max_iter=150, fuzzines=2, error=1e-5, random_state=None, dist="euclidean", method="Cmeans"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.fuzzines = fuzzines
        self.error = error
        self.random_state = random_state
        self.dist = dist
        self.method = method
        
    def fit(self, X):
        memberships = self._init_mem(X)
              
        iteration = 0
        while iteration < self.max_iter:
            membershipsNew = deepcopy(memberships)
            new_class_centers = self._update_centers(X, memberships)
            distance = self._calculate_dist(X,memberships,new_class_centers)
            memberships = self._update_memberships(X, memberships, new_class_centers, distance)
            iteration += 1
            if norm(memberships - membershipsNew) < self.error:
                break
            
        return memberships, new_class_centers
    
    def _init_mem(self,X):
        n_samples = X.shape[0]
        n_clusters = self.n_clusters

        #initialize memberships
        rnd = np.random.RandomState(self.random_state)
        memberships = rnd.rand(n_samples,n_clusters)

        #update membership relative to classes
        summation = memberships.sum(axis=1).reshape(-1,1)
        denominator = np.repeat(summation,n_clusters,axis=1)
        memberships = memberships/denominator
        
        return memberships

    def _update_centers(self, X, memberships):
        fuzzyMem = memberships ** self.fuzzines
        new_class_centers = (np.dot(X.T,fuzzyMem)/np.sum(fuzzyMem,axis=0)).T
        return new_class_centers
    
    def _calculate_fuzzyCov(self,X,memberships,new_class_centers):
        #calculating covariance matrix in its fuzzy form  
        fuzzyMem = memberships ** self.fuzzines
        n_clusters = self.n_clusters
        FcovInv_Class = []
        dim = X.shape[1]
        for i in range(n_clusters): 
            diff = X-new_class_centers[i]
            left = np.dot((fuzzyMem[:,i].reshape(-1,1)*diff).T,diff)/np.sum(fuzzyMem[:,i],axis=0)
            Fcov = (np.linalg.det(left)**(-1/dim))*left
            FcovInv = np.linalg.inv(Fcov)
            FcovInv_Class.append(FcovInv)
        return FcovInv_Class

    def _calculate_dist(self,X,memberships,new_class_centers):
        
        if self.method == "Gustafson–Kessel":
            n_clusters = self.n_clusters
            FcovInv_Class = self._calculate_fuzzyCov(X,memberships,new_class_centers)

            #calculating mahalanobis distance
            mahalanobis_Class = []
            for i in range(n_clusters): 
                diff = X-new_class_centers[i]
                left = np.dot(diff,FcovInv_Class[i])    
                mahalanobis = np.diag(np.dot(left,diff.T))
                mahalanobis_Class.append(mahalanobis)
            distance = np.array(mahalanobis_Class).T
            return distance
        
        elif self.method == "Cmeans":
            distance = cdist(X, new_class_centers,metric=self.dist)
            return distance

    def _update_memberships(self, X, memberships, new_class_centers, distance):
        fuzziness = self.fuzzines
        n_clusters = self.n_clusters
        n_samples = X.shape[0]
        
        power = float(2/(fuzziness - 1))
        distance = distance**power
        arr = np.zeros((n_samples,n_clusters))
        for i in range(n_clusters):
            for ii in range(n_clusters):
                arr[:,ii] = ((distance[:,i]/distance[:,ii]))
            memberships[:,i] = 1/np.sum(arr,axis=1)   
        return memberships
"""
import numpy as np
from scipy.linalg import norm

class GK:
    def __init__(self, n_clusters=4, max_iter=100, m=2, error=1e-6):
        super().__init__()
        self.u, self.centers, self.f = None, None, None
        self.clusters_count = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error

    def fit(self, z):
        N = z.shape[0]
        C = self.clusters_count
        centers = []

        u = np.random.dirichlet(np.ones(N), size=C)

        iteration = 0
        while iteration < self.max_iter:
            u2 = u.copy()

            centers = self.next_centers(z, u)
            f = self._covariance(z, centers, u)
            dist = self._distance(z, centers, f)
            u = self.next_u(dist)
            iteration += 1

            # Stopping rule
            if norm(u - u2) < self.error:
                break

        self.f = f
        self.u = u
        self.centers = centers
        return centers

    def next_centers(self, z, u):
        um = u ** self.m
        return ((um @ z).T / um.sum(axis=1)).T

    def _covariance(self, z, v, u):
        um = u ** self.m

        denominator = um.sum(axis=1).reshape(-1, 1, 1)
        temp = np.expand_dims(z.reshape(z.shape[0], 1, -1) - v.reshape(1, v.shape[0], -1), axis=3)
        temp = np.matmul(temp, temp.transpose((0, 1, 3, 2)))
        numerator = um.transpose().reshape(um.shape[1], um.shape[0], 1, 1) * temp
        numerator = numerator.sum(0)

        return numerator / denominator

    def _distance(self, z, v, f):
        dif = np.expand_dims(z.reshape(z.shape[0], 1, -1) - v.reshape(1, v.shape[0], -1), axis=3)
        determ = np.power(np.linalg.det(f), 1 / self.m)
        det_time_inv = determ.reshape(-1, 1, 1) * np.linalg.pinv(f)
        temp = np.matmul(dif.transpose((0, 1, 3, 2)), det_time_inv)
        output = np.matmul(temp, dif).squeeze().T
        return np.fmax(output, 1e-8)

    def next_u(self, d):
        power = float(1 / (self.m - 1))
        d = d.transpose()
        denominator_ = d.reshape((d.shape[0], 1, -1)).repeat(d.shape[-1], axis=1)
        denominator_ = np.power(d[:, None, :] / denominator_.transpose((0, 2, 1)), power)
        denominator_ = 1 / denominator_.sum(1)
        denominator_ = denominator_.transpose()

        return denominator_

    def predict(self, z):
        if len(z.shape) == 1:
            z = np.expand_dims(z, axis=0)

        dist = self._distance(z, self.centers, self.f)
        if len(dist.shape) == 1:
            dist = np.expand_dims(dist, axis=0)

        u = self.next_u(dist)
        return np.argmax(u, axis=0)