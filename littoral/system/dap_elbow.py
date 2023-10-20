from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
from collections import Counter

#######################################
# Optimizating the number of clusters #
#######################################

# Function Signature:
# function(X, ks, model, metric) -> int (k)

def crisp_elbow(X, ks, model, metric='distortion'):   
    max_iters = 100
    best_ks = []

    for _ in range(max_iters):
        visualizer = KElbowVisualizer(model, k=ks, metric=metric)
        visualizer.fit(X)
        best_ks.append(visualizer.elbow_value_)
        plt.clf()
    
    c = Counter(best_ks)
    return c.most_common()[0]

def fuzzy_elbow(X, ks, model, metric='distortion'):
    if(model.type == None or not model.type == 'fuzzy'):
        return -1

    return 1

#######################################