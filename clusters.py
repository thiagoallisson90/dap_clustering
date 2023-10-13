import numpy as np
import matplotlib.pyplot as plt

"""
- Code based on Implement Clustering Algorithms from Scratch
    * Link: https://www.kaggle.com/code/tatamikenn/implement-clustering-algorithms-from-scratch
"""

class CustomKMeans:
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
            self.distances = np.linalg.norm(self.data[:, None, :] - centroids, axis=2)
            labels = np.argmin(self.distances, axis=1)

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


class CustomKMedoids:
    def __init__(self, k, max_iter=300, verbose=False):
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

        for i in range(self.k):
            self.distances[:, i] = np.sqrt(np.sum(np.square(data - self.medoids[i]), axis=1))

        self.labels = np.argmin(self.distances, axis=1)
        self.old_labels = np.empty(self.N)
        self.all_idxs = np.arange(self.N)

        for it in range(self.max_iter):
            best_swap = (-1, -1, 0)
            best_distances = np.zeros(self.N)
            for i in range(self.k):
                non_medoids_idx = self.all_idxs[np.logical_not(np.isin(self.all_idxs, self.medoids_idx))]
                for j in non_medoids_idx:
                    new_medoid = self.data[j]
                    new_distances = np.sqrt(np.sum(np.square(self.data - new_medoid), axis=1))
                    cost_change = np.sum(new_distances[self.labels == i]) - np.sum(self.distances[self.labels == i, i])
                    if cost_change < best_swap[2]:
                        best_swap = (i, j, cost_change)
                        best_distances = new_distances
            if best_swap == (-1, -1, 0):
                break

            i, j, _ = best_swap
            self.distances[:, i] = best_distances
            self.medoids[i] = self.data[j]

            self.labels = np.argmin(self.distances, axis=1)
            self.medoid_history.append(self.medoids.copy())

            self.old_labels = self.labels
        if(self.verbose):
            print(f'* converged after {it + 1} iterations')
        self.medoid_history = np.array(self.medoid_history)

        for i in range(self.k):
            self.clusters[i] = self.data[self.labels == i]

    def get_clusters(self):
        return self.labels, self.medoids, self.medoid_history, self.clusters

if __name__ == '__main__':
    from sklearn_extra.cluster import KMedoids
    from test_cluster import generate_ed_coords
    from scipy.spatial.distance import euclidean
    from yellowbrick.cluster import KElbowVisualizer
    from sklearn.metrics import calinski_harabasz_score

    max_dist = 2000
    max_clusters = 30
    coords = generate_ed_coords()
    ks = []

    for k in range(1, max_clusters+1):
        model = KMedoids(k)
        model.fit(coords)

        centroids = model.cluster_centers_
        clusters = [[] for _ in range(k)]
        insert = True
        for i, label in enumerate(model.labels_):
            clusters[label].append(euclidean(coords[i], centroids[label]))
            if(max(clusters[label]) > max_dist):
                print(k, label, max(clusters[label]))
                insert = False
                break
        
        if(insert):
            ks.append(k)
    
    print(ks)

    elbows = []
    scores = []
    for i in range(30):
        vis = KElbowVisualizer(KMedoids(), metric='silhouette', k=ks, timings=False)
        vis.fit(coords)
        elbows.append(vis.elbow_value_)
        scores.append(vis.elbow_score_)
        plt.clf()

    idx = np.argmax(scores)
    print(idx, elbows[idx], scores[idx])
    
    model = KMedoids(elbows[idx])
    model.fit(coords)
    score = calinski_harabasz_score(coords, model.labels_)
    print(f'score = {score}')

    centroids = model.cluster_centers_
    clusters = [[] for _ in range(elbows[idx])]
    for i, label in enumerate(model.labels_):
        clusters[label].append(euclidean(coords[i], centroids[label]))

    for cluster in clusters:
        print(max(cluster))


