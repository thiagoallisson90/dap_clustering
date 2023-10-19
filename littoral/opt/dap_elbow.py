from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from collections import Counter

#######################################
# Optimizating the number of clusters #
#######################################

def elbow(X, ks, model, metric='distortion'):
    max_iters = 100
    best_ks = []

    for _ in range(max_iters):
        visualizer = KElbowVisualizer(model, k=ks, metric=metric)
        visualizer.fit(X)
        best_ks.append(visualizer.elbow_value_)
        plt.clf()
    
    c = Counter(best_ks)
    print(c)
    return c.most_common()[0]

#######################################