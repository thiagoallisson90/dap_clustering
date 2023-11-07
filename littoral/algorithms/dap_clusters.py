import numpy as np

from littoral.system.dap_simulate import simulate_tests

"""
- Code based on Implement Clustering Algorithms from Scratch
    * Link: https://www.kaggle.com/code/tatamikenn/implement-clustering-algorithms-from-scratch
"""

#############################
# K-Means based on the RSSI #
#############################
class DapKMeans:
    def __init__(self, k, max_iter=300, verbose=False):
        self.k = k
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, data):
        self.data = data  
        N = data.shape[0]
        self.distances = np.zeros((N, self.k))
        centroids = data[np.random.choice(N, self.k, replace=False), :]
        cluster_center_history = [centroids.copy()]
        self.clusters = [[] for _ in range(self.k)]

        labels = np.empty(N)
        old_labels = np.empty(N)

        for i in range(self.max_iter):
            self.distances = simulate_tests(self.data, centroids, 'kmeans_rssi')
            labels = np.argmax(self.distances, axis=1)

            for j in range(self.k):
                centroids[j] = np.mean(self.data[labels == j], axis=0)
            cluster_center_history.append(centroids.copy())

            if i > 0 and np.all(labels == old_labels):
                break

            old_labels = labels
        
        if(self.verbose):
            print(f'* converged after {i + 1} iterations')

        self.labels = labels
        self.centroids = centroids
        self.cluster_center_history = np.array(cluster_center_history)
        
        for i in range(self.k):
            self.clusters[i] = self.data[self.labels == i]

    def get_clusters(self):
        return self.labels, self.centroids, self.cluster_center_history, self.clusters

###############################
# K-Medoids based on the RSSI #
###############################
class DapKMedoids:
    def __init__(self, k, max_iter=100, verbose=False):
        self.k = k
        self.max_iter = max_iter
        self.verbose = verbose

    def fit(self, data):
        self.data = data
        self.N = data.shape[0]
        self.medoids_idx = np.random.choice(self.N, self.k, replace=False)
        self.medoids = data[self.medoids_idx].copy()
        self.distances = np.zeros((self.N, self.k))
        self.medoid_history = [self.medoids.copy()]
        self.clusters = [[] for _ in range(self.k)]
        self.costs = 0

        self.distances = simulate_tests(self.data, self.medoids, 'kmedoids_rssi')
        self.labels = np.argmax(self.distances, axis=1)
        
        for j, label in enumerate(self.labels):
            self.costs = self.costs + self.distances[j][label]

        self.old_labels = np.empty(self.N)
        self.all_idxs = np.arange(self.N)

        for it in range(self.max_iter):
            for i in range(self.k):
                non_medoids_idx = self.all_idxs[np.logical_not(np.isin(self.all_idxs, self.medoids_idx))]
                
                for j in non_medoids_idx:
                    new_medoids = self.medoids.copy()
                    new_medoids[i] = self.data[j]
                    new_distances = simulate_tests(self.data, new_medoids, 'kmedoids_rssi')
                    new_labels = np.argmax(new_distances, axis=1)
            
                    new_costs = []            
                    for j, label in enumerate(new_labels):
                        new_costs = new_costs + new_distances[j][label]

                    if(new_costs > self.costs):
                        self.old_labels = self.labels
                        self.medoid_history.append(self.medoids.copy())

                        self.medoids = new_medoids
                        self.distances = new_distances
                        self.costs = new_costs
                        self.labels = new_labels
                                
        if(self.verbose):
            print(f'* converged after {it + 1} iterations')

        self.medoid_history = np.array(self.medoid_history)

        for i in range(self.k):
            self.clusters[i] = self.data[self.labels == i]

    def get_clusters(self):
        return self.labels, self.medoids, self.medoid_history, self.clusters
