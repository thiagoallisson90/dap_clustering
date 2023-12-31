from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import distortion_score
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from kneed import KneeLocator
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

#######################################
# Optimizating the number of clusters #
#######################################

# Function Signature:
# function(X, ks, model, metric) -> int (k)

method = {
    'distortion': distortion_score,
    'silhouette': silhouette_score,
    'calinski_harabasz': calinski_harabasz_score,
}

direction = {
    'distortion': 'decreasing',
    'silhouette': 'increasing',
    'calinski_harabasz': 'increasing',
}

class Elbow:
    pass

class CrispElbow(Elbow):
    def execute(self, X, ks, model, metric='distortion'):
        max_iters = 100
        best_ks = []

        for _ in range(max_iters):
            visualizer = KElbowVisualizer(model, k=ks, metric=metric)
            visualizer.fit(X)
            best_ks.append(visualizer.elbow_value_)
            plt.clf()
        
        c = Counter(best_ks)
        return c.most_common()[0]

class CMeansElbow(Elbow):
    def execute(self, X, ks, metric='distortion'):
        from skfuzzy.cluster import cmeans

        found_knees = []
        n_iters = 100

        for _ in range(n_iters):
            distortion_scores = []
            k_values = ks
            for k in k_values:
                _, u, _, _, _, _, _ = cmeans(X.T, c=k, m=2, error=0.005, maxiter=1000)
                labels = np.argmax(u, axis=0)
                distortion_scores.append(method[metric](X, labels))
                
            kl = KneeLocator(x=k_values, 
                            y=distortion_scores, 
                            curve='convex', 
                            direction=direction[metric],
                            S=1
                            )
            
            found_knees.append(kl.knee) 

        return Counter(found_knees).most_common()[0]

class GKElbow(Elbow):
    def execute(self, X, ks, metric='distortion'):
        from littoral.algorithms.dap_gk import GK
        
        found_knees = []
        n_iters = 100

        for _ in range(n_iters):
            distortion_scores = []
            k_values = ks
            for k in k_values:
                clf = GK(k, m=2)
                clf.fit(X)
                labels = np.argmax(clf.u, axis=0)
                distortion_score.append(method[metric](X, labels))
                
            kl = KneeLocator(x=k_values, 
                            y=distortion_scores, 
                            curve='convex', 
                            direction=direction[metric],
                            S=1
                            )
            
            found_knees.append(kl.knee) 

        return Counter(found_knees).most_common()[0]


#######################################