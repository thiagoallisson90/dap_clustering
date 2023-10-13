import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from skfuzzy.cluster import cmeans

def optimalK(data, nrefs=3, maxClusters=15, model_name='kmeans'):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = []
    for gap_index, k in enumerate(range(1, maxClusters)):

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


def optimalC1(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = []
    for gap_index, k in enumerate(range(1, maxClusters)):

        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)

        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            cntr, u, u0, d, jm, p, fpc = cmeans(data=randomReference, c=k, m=2, error=0.005, maxiter=1000)
            
            refDisps[i] = jm.mean()

        # Fit cluster to original data and create dispersion
        cntr, u, u0, d, jm, p, fpc = cmeans(data=data, c=k, m=2, error=0.005, maxiter=1000)
        
        origDisp = jm.mean()

        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)

        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf.append({'clusterCount':k, 'gap':gap})

    # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal
    return (gaps.argmax() + 1), pd.DataFrame(resultsdf)


def range_limit(d, max=2000):
    df = pd.DataFrame(d)
    for column in df.columns[0:]:
        for item in df[column]:
            if item >= max:
                df.replace(item, 10000, True)
    return np.array(df)


def points_limit(data, k, maxPoints=100):
    desorganized = True
    while desorganized:
        cntr, u, u0, d, jm, p, fpc = cmeans(data=data, c=k, m=2, error=0.005, maxiter=1000, init=None)
        df = pd.DataFrame(u)
        a = 0
        for _, row in df.iterrows():
            aux = []
            for item in row:
                if item >= 0.6:
                    aux.append(item)
            if len(aux) >= maxPoints:
                k += 1
                break
            else:
                a += 1
        if a == k:
            desorganized = False
    return k

def calculate_dispersions(c, randomReference):
        cntr, u, u0, d, jm, p, fpc = cmeans(data=randomReference, c=c, m=2, error=0.005, maxiter=1000)
        BWkb = np.log(np.mean(jm))
        return BWkb

def optimalC2(data, nrefs=3, maxClusters=15):
    """
    Calculates Fuzzy C-Means optimal C using Gap Statistic from Sentelle, Hong, Georgiopoulos, Anagnostopoulos
    Params:
        data: ndarray of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (k, resultsdf, gp)
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

    num_cores = joblib.cpu_count()  # Adjust this to control the number of CPU cores to use in parallel computations

    for gap_index, c in enumerate(range(1, maxClusters)):
        # Use joblib to calculate reference dispersions in parallel
        BWkbs = joblib.Parallel(n_jobs=-1)(
            joblib.delayed(calculate_dispersions)(c, np.random.random_sample(size=data.shape)) for i in range(nrefs)
        )

        cntr, u, u0, d, jm, p, fpc = cmeans(data=data, c=c, m=2, error=0.005, maxiter=1000)

        Wks[gap_index] = np.log(np.mean(jm))

        gap = sum(BWkbs) / nrefs - Wks[gap_index]

        gaps[gap_index] = gap

        resultsdf.append({'clusterCount': c, 'gap': gap})

        Wkbs[gap_index] = sum(BWkbs) / nrefs
        sdk[gap_index] = np.sqrt((sum((BWkbs - Wkbs[gap_index])**2)) / nrefs)

        sError[gap_index] = sdk[gap_index] * np.sqrt(1 + 1 / nrefs)

    for k, _ in enumerate(range(1, maxClusters)):
        if not k == len(gaps) - 1:
            if gaps[k] >= gaps[k + 1] - sError[k + 1]:
                gaps_sks[k] = gaps[k] - gaps[k + 1] - sError[k + 1]
        else:
            gaps_sks[k] = -20

        gp.append({'clusterCount': k+1, 'Gap_sk': gaps_sks[k]})

    iter_points = [x[0]+1 for x in sorted([y for y in enumerate(gaps)], key=lambda x: x[1], reverse=True)[:3]]
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