import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from skfuzzy.cluster import cmeans

def optimalK(data, nrefs=3, min_clusters = 1, maxClusters=15, model_name='kmeans'):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(min_clusters, maxClusters)),))
    resultsdf = []
    for gap_index, k in enumerate(range(min_clusters, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k, n_init='auto') if model_name == 'kmeans' else KMedoids(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp

        # Fit cluster to original data and create dispersion
        km = KMeans(k, n_init='auto') if model_name == 'kmeans' else KMedoids(k)
        km.fit(data)
        
        origDisp = km.inertia_

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf.append({'clusterCount':k, 'gap':gap})

    # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
    return (gaps.argmax() + 1), pd.DataFrame(resultsdf)

def optimalC(data, nrefs=3, maxClusters=15):
    """
    Calculates Fuzzy C-Means optimal C using Gap Statistic from Sentelle, Hong, Georgiopoulos, Anagnostopoulos
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalC)
    """
    sdk = np.zeros(len(range(0, maxClusters)))
    sError = np.zeros(len(range(0, maxClusters)))
    BWkbs = np.zeros(len(range(0, nrefs)))
    Wks = np.zeros(len(range(0, maxClusters)))
    Wkbs = np.zeros(len(range(0, maxClusters)))
    gaps_sks = np.zeros((len(range(1, maxClusters))))
    gp = []

    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = []

    for gap_index, c in enumerate(range(1, maxClusters)):

        # For n references, generate random sample and perform cmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            cntr, u, u0, d, jm, p, fpc = cmeans(data=randomReference, c=c, m=2, error=0.005, maxiter=1000)

            # Holder for reference dispersion results
            BWkbs[i] = np.log(jm[len(jm)-1])
            #BWkbs[i] = np.log(np.mean(jm))

        # Fit cluster to original data and create dispersion
        cntr, u, u0, d, jm, p, fpc = cmeans(data=data, c=c, m=2, error=0.005, maxiter=1000)

        # Holder for original dispersion results
        Wks[gap_index] = np.log(jm[len(jm)-1])
        #Wks[gap_index] = np.log(np.mean(jm))

        # Calculate gap statistic
        gap = sum(BWkbs)/nrefs - Wks[gap_index]

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        # Assign gap values and cluster count to plot curve graph
        resultsdf.append({'clusterCount': c, 'gap': gap})

        # Compute Standard Deviation
        Wkbs[gap_index] = sum(BWkbs) / nrefs
        sdk[gap_index] = np.sqrt((sum((BWkbs - Wkbs[gap_index])**2))/nrefs)

        # Compute the Simulation Error
        sError[gap_index] = sdk[gap_index] * np.sqrt(1 + 1 / nrefs)

    for k, _ in enumerate(range(1, maxClusters)):
        if not k == len(gaps) - 1:
            if gaps[k] >= gaps[k + 1] - sError[k + 1]:
                gaps_sks[k] = gaps[k] - gaps[k + 1] - sError[k + 1]
        else:
            gaps_sks[k] = -20

        # Assign new gap values calculated by simulation error and cluster count to plot bar graph
        gp.append({'clusterCount': k+1, 'Gap_sk': gaps_sks[k]})

    # Assign best cluster numbers by gap values
    iter_points = [x[0]+1 for x in sorted([y for y in enumerate(gaps)], key=lambda x: x[1], reverse=True)[:3]]

    # Assign best cluster numbers by gap values calculated with simulation error
    iter_points_sk = [x[0]+1 for x in sorted([y for y in enumerate(gaps_sks)], key=lambda x: x[1], reverse=True)[:3]]

    a = list(filter(lambda g: g in iter_points, iter_points_sk))
    if a:
        if not min(a) == 1:
            k = min(a)
        else:
            a.remove(1)
            k = min(a)
    else:
        k = min(iter_points_sk)

    return k, pd.DataFrame(resultsdf), pd.DataFrame(gp)


def optimalC1(data, nrefs=3, min_clusters=10, maxClusters=30):
    gaps = np.zeros((len(range(min_clusters, maxClusters)),))
    resultsdf = []

    best_gap = None
    best_k = -1
    gap_index = 0
    for k in range(min_clusters, maxClusters):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            _, _, _, _, jm, _, _ = cmeans(data=randomReference, c=k, m=2, error=0.005, maxiter=1000)
            refDisps[i] = jm.mean()

        # Fit cluster to original data and create dispersion
        _, _, _, _, jm, _, _ = cmeans(data=data, c=k, m=2, error=0.005, maxiter=1000)
        origDisp = jm.mean()

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        gap_index += 1

        if(best_gap == None or best_gap < gap):
            best_gap = gap
            best_k =  k
        
        resultsdf.append({'clusterCount':k, 'gap':gap})

    # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
    return best_k, pd.DataFrame(resultsdf), best_gap

