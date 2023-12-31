import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from gapstatistic import optimalK, optimalC1
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from fcluster import FCluster
from skfuzzy.cluster import cmeans
from yellowbrick.cluster import distortion_score
from kneed import KneeLocator
from collections import Counter

####################
# Global Variables #
####################
scratch_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch'
dap = 'dap_clustering'
base_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering'
img_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering/imgs'
data_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering/data'
ed_pos_file = 'ed_pos_file.csv'
ed_out_file = 'ed_out_file.csv'
gw_pos_file = 'gw_pos_file.csv'
ns3_cmd = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/./ns3'
####################

#####################
# General Functions #
#####################
def generate_ed_coords(n_points=2000, axis_range = 10000, seed=42):
  np.random.seed(seed)
  x = np.random.uniform(0, axis_range, n_points)
  y = np.random.uniform(0, axis_range, n_points)

  np.random.seed(None)
  
  return np.column_stack((x, y))

def write_coords(data, filename):
    with open(f'{base_dir}/{filename}', mode='w') as file:
        for d in data:
            file.write(f'{d[0]},{d[1]}\n')

def simulate(coords, centroids, model, ed_pos_file=ed_pos_file, ed_out_file=ed_out_file, 
             gw_pos_file=gw_pos_file, radius=10000, load=1):
    script='scratch/dap_clustering.cc'
    n_gw = len(centroids)
    n_simulatons = 30
    
    write_coords(coords, ed_pos_file)
    write_coords(centroids, gw_pos_file)
    
    with open(f'{data_dir}/{model}/tracker_{load}_unconfirmed_buildings{n_gw}gw.csv', mode="w") as file:
        file.write('')
        file.close()

    params01 = f'--edPos={ed_pos_file} --edOutputFile={ed_out_file} --gwPos={gw_pos_file} --nGateways={n_gw}'
    params02 = f'--cModel={model} --radius={radius} --nDevices={len(coords)} --lambda={load}'

    for i in range(1, n_simulatons+1):
        run_cmd = \
            f'{ns3_cmd} run "{script} {params01} {params02} --nRun={i}"'
        os.system(run_cmd)

def write_scores(data, filename):
    with open(f'{base_dir}/{filename}', mode='a') as file:
        file.write(f'{data["k"]},{data["score"]}\n')

def capex_opex_calc(n_gws):
    CBs, Cins, Cset, Txinst = 1, 2, 0.1, 4

    Cman = 0.125
    Clease, Celet, Ctrans, t = 1, 1, 0.1, 1

    capex = n_gws * (CBs + Cins + Cset + Txinst)
    #opex = (Cman*capex + n_gws * (Clease + Celet + Ctrans)) * t
    opex = (Cman*capex + n_gws) * t

    return capex, opex
####################

##############
# Clustering #
##############
def clustering(X, n_clusters, labels):
    cluster_points = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        cluster_points[label].append(X[i])
    
    return cluster_points

def cm_cluster(X, n_clusters=2, m=2, error=0.005, max_iter=1000):
    cntr, u, _, _, _, _, _ = cmeans(X.T, c=n_clusters, m=m, error=error, maxiter=max_iter)
    labels = np.argmax(u, axis=0)    
    cluster_points = clustering(X, n_clusters, labels)

    return cntr, labels, cluster_points

def gk_cluster(X, n_clusters=2, m=2, error=1e-5, max_iter=150):
    clf = FCluster(n_clusters=n_clusters, method='Gustafson–Kessel', fuzzines=m, error=error, max_iter=max_iter)
    u, cntr = clf.fit(X)
    labels = np.argmax(u, axis=-1)
    cluster_points = clustering(X, n_clusters, labels)
    
    return cntr, labels, cluster_points

def kmeans_cluster(X, n_clusters=2):
    clf = KMeans(n_clusters, n_init='auto', init='k-means++').fit(X)
    cluster_points = clustering(X, n_clusters, clf.labels_)

    return clf.cluster_centers_, clf.labels_, cluster_points

def kmedoids_cluster(X, n_clusters=2):
    clf = KMedoids(n_clusters).fit(X)
    cluster_points = clustering(X, n_clusters, clf.labels_)

    return clf.cluster_centers_, clf.labels_, cluster_points

def rand_cluster(X, n_cluster=16):
    cluster_centers = generate_ed_coords(n_cluster, seed=None)

run_model = {
    'kmeans': kmeans_cluster,
    'kmedoids':  kmedoids_cluster,
    'cmeans': cm_cluster,
    'gk': gk_cluster,
    'rand': rand_cluster,
}

def elbow(X, k_lims, model='kmeans'):
    found_knees = []
    n_iters = 100
    for i in range(n_iters):
        distortion_scores = []
        k_values = range(k_lims[0], k_lims[1]+1)
        for k in k_values:
            _, labels, _ = run_model[model](X, k)
            distortion_scores.append(distortion_score(X, labels))
            
        kl = KneeLocator(x=k_values, 
                        y=distortion_scores, 
                        curve='convex', 
                        direction='decreasing', 
                        S=1
                        )
        
        found_knees.append(kl.knee) 

    return Counter(found_knees).most_common()[0]
###########################

###########################
# Clustering Optimization #
###########################
def opt_kmodel(data, model_name='kmeans', metric_name='gap'):
    from sklearn.cluster import KMeans
    from sklearn_extra.cluster import KMedoids

    models = {
        'kmeans': KMeans(n_init='auto'),
        'kmedoids': KMedoids(),
    }

    metrics = {
        'elbow': 'distortion',
        'silhouette': 'silhouette',
        'calinski': 'calinski_harabasz',
        #'gap': 'gap_statistic'
    }

    print(f'Running ({model_name},{metric_name})')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    #min_clusters = 1 if metric_name in ['elbow', 'gap'] else 2
    min_clusters = 10
    max_clusters = 30
    num_iter = 30
    n_refs = 500 if model_name == 'kmeans' else 50

    opt_dir = f'{data_dir}/{model_name}/{metric_name}'

    scores = []

    if not metric_name == 'gap':
        for i in range(num_iter): 
            visualizer = KElbowVisualizer(models[model_name], metric=metrics[metric_name],
                                          k=(min_clusters, max_clusters+1), timings=False, n_jobs=-1)    
            visualizer.fit(data)
            
            k = visualizer.elbow_value_
            score = visualizer.elbow_score_

            #write_scores({'k': k, 'score': score}, f'{opt_dir}/daps.csv')
            scores.append({'k': k, 'score': score})
            
            print(f'k defined in the iteration {i+1} with {metric_name} method = {k}')
            plt.clf()
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    else:
        for i in range(num_iter-10):
            k, gap_history = optimalK(data, nrefs=n_refs, min_clusters=10, maxClusters=max_clusters+1, model_name=model_name)            
            score = gap_history['gap'][k-1]

            #write_scores({'k': k, 'score': score}, f'{opt_dir}/daps.csv')
            scores.append({'k': k, 'score': score})
            
            print(f'k defined in the iteration {i+1} with {metric_name} = {k}')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')    
    
    pd.DataFrame(scores).to_csv(f'{opt_dir}/daps.csv', index=False, header=False)

def opt_model():
    models_name = ['kmeans', 'kmedoids']
    #metric_names = ['elbow', 'silhouette', 'calinski', 'gap']
    metric_names = ['elbow']

    write_coords(X, ed_pos_file)

    for model in models_name:
        for metric in metric_names:
            opt_kmodel(X, model, metric)

def opt_cmeans():
    ed_coords = generate_ed_coords() 

    num_iters = 20
    max_clusters = 30

    scores = []

    print('Running (cmeans,gap)')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for i in range(num_iters):
        k, results, gap = optimalC1(ed_coords.T, nrefs=20, min_clusters=10, maxClusters=max_clusters+1)

        #write_scores({'k': k, 'score': score}, f'data/cmeans/gap/daps.csv')
        scores.append({'k': k, 'score': gap})

        print(f'k defined in the iteration {i+1} with gap statistics = {gap} is {k}')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')    
    
    pd.DataFrame(scores).to_csv(f'{data_dir}/cmeans/gap/daps.csv', index=False, header=False)

def opt_models(X):
    models_name = ['kmeans', 'kmedoids', 'cmeans', 'gk']

    write_coords(X, ed_pos_file)
    
    ks = {}
    scores = {}
    for model in models_name:
        k, score = elbow(X, [10, 30], model)
        ks[model] = k
        scores[model] = score

    return ks, scores
###################

######################
# Plotting Functions #
######################
def plot_clusters(X, k, model='kmeans'):
    title_model = {
        'kmeans': 'K-Means',
        'kmedoids': 'K-Medoids',
        'cmeans':  'Fuzzy C-Means',
        'gk': 'Gustafson-Kessel Clustering',
        'rand16': 'Rand16',
        'rand25': 'Rand25',
    }
    cntr, labels, cluster_points = run_model[model](X, k)

    colors = plt.cm.rainbow(np.linspace(0, 1, k))

    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', alpha=0.7, s=100)

    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'{title_model[model]}', fontsize=16)

    image_path = f'{img_dir}/antenna.jpg'
    image = plt.imread(image_path)
    for i, centroid in enumerate(cntr):
        imagebox = OffsetImage(image, zoom=0.1)
        ab = AnnotationBbox(imagebox, centroid, frameon=False)
        plt.gca().add_artist(ab)
        plt.text(centroid[0] + 1, centroid[1], f'     {i+1}', fontsize=16, fontweight="bold")

    plt.savefig(f'{img_dir}/{model}/{k}gw.png')
    plt.clf()

    cluster_counts = np.bincount(labels)

    sorted_indices = np.argsort(cluster_counts)[::-1]
    sorted_cluster_counts = cluster_counts[sorted_indices]
    sorted_cluster_labels = [f'Cluster {i+1}' for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    
    plt.barh(range(k), sorted_cluster_counts, tick_label=sorted_cluster_labels, color=colors)
    plt.xlabel('SMs per Cluster', fontsize=14)
    plt.ylabel('Clusters', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title('Number of SMs in each Cluster', fontsize=16)
    plt.savefig(f'{img_dir}/{model}/{k}gw_chart.png', bbox_inches='tight')
    plt.clf()

    intra_cluster_distances = {}
    for label, points in enumerate(cluster_points):
        distances = pairwise_distances(points, metric='euclidean')
        intra_cluster_distances[label] = distances

    average_intra_cluster_distances = {}
    max_intra_cluster_distances = {}
    for label, distances in intra_cluster_distances.items():
        average_distance = np.mean(distances)
        max_distance = np.max(distances)
        average_intra_cluster_distances[label] = average_distance
        max_intra_cluster_distances[label] = max_distance
    
    data = {
        'cluster': list(range(1, k+1)),
        'mean_dist': [round(d, 2) for d in average_intra_cluster_distances.values()],
        'max_dist': [round(d, 2) for d in max_intra_cluster_distances.values()]
    }

    write_coords(cntr, f'data/{model}/{k}gw_centroids.csv')
    pd.DataFrame(data).to_csv(f'{data_dir}/{model}/{k}gw_distances.csv', index=False)

    return cntr, labels, cluster_points

def plot_copex(ks):
    n_gws_kmeans, n_gws_kmedoids, n_gws_cmeans, n_gws_gk = \
        ks['kmeans'], ks['kmedoids'], ks['cmeans'], ks['gk']
    
    capex1, opex1 = capex_opex_calc(n_gws_kmeans)
    capex2, opex2 = capex_opex_calc(n_gws_kmedoids)
    capex3, opex3 = capex_opex_calc(n_gws_cmeans)
    capex4, opex4 = capex_opex_calc(n_gws_gk)

    capex_values = [capex1, capex2, capex3, capex4]
    opex_values = [opex1, opex2, opex3, opex4]

    scenario_labels = ['K-Means', 'K-Medoids', 'C-Means', 'Gustafson-Kessel']
    #colors = ['red', 'green', 'blue', 'orange']
    colors = ['0.2', '0.4', '0.6', '0.8']
    hatches = ['//', '--', '++', 'x']
    text = 'K-Means, K-Medoids, C-Means and Gustafson-Kessel'

    plt.figure(figsize=(12, 8))
    plt.bar(scenario_labels, capex_values, color=colors, hatch=hatches)
    plt.xlabel('Clustering Models', fontsize=14)
    plt.ylabel('CapEx (K€)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'CapEx - {text}', fontsize=16)

    for i, value in enumerate(capex_values):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    plt.savefig(f'{img_dir}/capex_models.png')
    plt.clf()

    plt.figure(figsize=(12, 8))
    plt.bar(scenario_labels, opex_values, color=colors, hatch=hatches)
    plt.xlabel('Clustering Models', fontsize=14)
    plt.ylabel('OpEx (K€)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'OpEx - {text}', fontsize=16)
    
    for i, value in enumerate(opex_values):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.savefig(f'{img_dir}/opex_models.png')
    plt.clf()

def plot_dists(ks):
    kmeans_dists = pd.read_csv(f'{data_dir}/kmeans/{ks["kmeans"]}gw_distances.csv')
    kmedoids_dists = pd.read_csv(f'{data_dir}/kmedoids/{ks["kmedoids"]}gw_distances.csv')
    cmeans_dists = pd.read_csv(f'{data_dir}/cmeans/{ks["cmeans"]}gw_distances.csv')
    gk_dists = pd.read_csv(f'{data_dir}/gk/{ks["gk"]}gw_distances.csv')

    models = ['K-Means', 'K-Medoids', 'Fuzzy C-Means', 'Gustafson-Kessel']
    mean_dist = [kmeans_dists['mean_dist'].mean(), kmedoids_dists['mean_dist'].mean(), 
                 cmeans_dists['mean_dist'].mean(), gk_dists['mean_dist'].mean()]
    max_dist = [kmeans_dists['max_dist'].mean(), kmedoids_dists['max_dist'].mean(), 
                cmeans_dists['max_dist'].mean(), gk_dists['max_dist'].mean()]

    width = 0.35
    x = np.arange(len(models))

    plt.figure(figsize=(12, 8))
    plt.bar(x, mean_dist, width, label='Mean Distance', color='0.3', hatch='//')
    plt.bar(x + width, max_dist, width, label='Maximium Distance', color='0.6', hatch='x')

    plt.xlabel('Clustering Algorithms', fontsize=14)
    plt.ylabel('Distances (m)', fontsize=14)
    plt.title('Intra-cluster Distances', fontsize=16)
    plt.xticks(x + width / 2, models, fontsize=12)
    plt.yticks(fontsize=12)

    for i, (mean_value, max_value) in enumerate(zip(mean_dist, max_dist)):
        plt.text(x[i], mean_value, f'{mean_value:.2f}', ha='center', va='bottom', fontsize=12)
        plt.text(x[i] + width, max_value, f'{max_value:.2f}', ha='center', va='bottom', fontsize=12)

    plt.legend()

    plt.savefig(f'{img_dir}/dists_models.png')
    plt.clf()

def plot_metrics(ks):
    import seaborn as sns

    n_gws_kmeans, n_gws_kmedoids, n_gws_cmeans, n_gws_gk = ks['kmeans'], ks['kmedoids'], ks['cmeans'], ks['gk']

    names=['sent', 'received', 'ul-pdr', 'rssi', 'snr', 'delay']
    df_partial_kmeans = \
        pd.read_csv(f'{data_dir}/kmeans/tracker_1_unconfirmed_buildings{n_gws_kmeans}gw.csv', names=names)

    df_partial_kmedoids = \
        pd.read_csv(f'{data_dir}/kmedoids/tracker_1_unconfirmed_buildings{n_gws_kmedoids}gw.csv', names=names)
    
    df_partial_cmeans = \
        pd.read_csv(f'{data_dir}/cmeans/tracker_1_unconfirmed_buildings{n_gws_cmeans}gw.csv', names=names)
    
    df_partial_cmeans = \
        pd.read_csv(f'{data_dir}/gk/tracker_1_unconfirmed_buildings{n_gws_gk}gw.csv', names=names)

    labels = ['K-Means', 'K-Medoids', 'Fuzzy C-Means', 'Gustafson-Kessel']
    metrics = ['ul-pdr', 'rssi', 'snr', 'delay']
    text = 'K-Means, K-Medoids, Fuzzy C-Means and Gustafson-Kessel'
    colors = ['0.2', '0.4', '0.6', '0.8']
    hatches = ['//', '--', '++', 'x']

    units = {
        'ul-pdr': ' (%)',
        'rssi': ' (dBm)',
        'snr': ' (dB)',
        'delay': ' (s)',
    }
    for metric in metrics:
        datas = {
            'K-Means': df_partial_kmeans[metric], 
            'K-Medoids': df_partial_kmedoids[metric],
            'Fuzzy C-Means': df_partial_cmeans[metric],
            'Gustafson-Kessel': df_partial_cmeans[metric]
        }
        upper_metric = metric.upper()

        plt.figure(figsize=(12, 8))
        
        ax = sns.boxplot(pd.DataFrame(datas), width=0.5, palette=colors)
        for i, patch in enumerate(ax.patches):
            patch.set_hatch(hatches[i])

        plt.xlabel('Clustering Models', fontsize=14)
        plt.ylabel(upper_metric + units[metric], fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(f'{upper_metric} - {text}', fontsize=16)
        plt.legend(labels=labels, title='Clustering Models', fontsize=10)
        
        plt.savefig(f'{img_dir}/{metric}_models.png')
        plt.clf()
    
    init_energy = 10000 # in J
    names = ['uid', 'remainder_energy']
    n_reps = 30
    kmeans_energy = \
        [pd.read_csv(f'{data_dir}/kmeans/nRun_{i}_{n_gws_kmeans}gws_battery-level.txt', names=names).drop(0, axis=0) \
            for i in range(1, n_reps+1)]
    
    kmedoids_energy = \
        [pd.read_csv(f'{data_dir}/kmedoids/nRun_{i}_{n_gws_kmedoids}gws_battery-level.txt', names=names).drop(0, axis=0) \
            for i in range(1, n_reps+1)]
    
    cmeans_energy = \
        [pd.read_csv(f'{data_dir}/cmeans/nRun_{i}_{n_gws_cmeans}gws_battery-level.txt', names=names).drop(0, axis=0) \
            for i in range(1, n_reps+1)]

    gk_energy = \
        [pd.read_csv(f'{data_dir}/gk/nRun_{i}_{n_gws_gk}gws_battery-level.txt', names=names).drop(0, axis=0) \
            for i in range(1, n_reps+1)]

    datas = {
        'K-Means': [(init_energy - df['remainder_energy'].mean()) for df in kmeans_energy],
        'K-Medoids': [(init_energy - df['remainder_energy'].mean()) for df in kmedoids_energy],
        'C-Means': [(init_energy - df['remainder_energy'].mean()) for df in cmeans_energy],
        'Gustafson-Kessel': [(init_energy - df['remainder_energy'].mean()) for df in gk_energy]
    }

    plt.figure(figsize=(12, 8))
    
    ax = sns.boxplot(pd.DataFrame(datas), width=0.5, palette=colors)
    for i, patch in enumerate(ax.patches):
        patch.set_hatch(hatches[i])

    plt.xlabel('Clustering Models', fontsize=14)
    plt.ylabel('Consumed Energy (J)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f'Consumed Energy - {text}', fontsize=16)
    plt.legend(labels=labels, title='Clustering Models', fontsize=10)
    
    plt.savefig(f'{img_dir}/energy_models.png')
    plt.clf()
##################

########
# Main #
########
def simulate_models():
    from sklearn.cluster import KMeans
    from sklearn_extra.cluster import KMedoids

    ed_coords = generate_ed_coords()

    model_names = ['kmeans', 'kmedoids']
    models = {
        'kmeans': lambda k: KMeans(k, n_init='auto'),
        'kmedoids': lambda k: KMedoids(k),
    }
    metric_names = ['elbow']
    ks = {}

    for model in model_names:
        for metric in metric_names:
            df = pd.read_csv(f'{base_dir}/data/{model}/{metric}/daps.csv', names=['k', 'score'])
            k = df.loc[df['score'].idxmin(), 'k']
            ks[model] = k
            clf = models[model](ks[model])
            clf.fit(ed_coords)
            centroids = clf.cluster_centers_
            plot_clusters(ed_coords, clf, model)
            simulate(ed_coords, centroids, model)

    # Simulating C-Means
    df = pd.read_csv(f'{data_dir}/cmeans/gap/daps.csv', names=['k', 'gap'])
    k = df.loc[df['gap'].idxmax(), 'k']
    ks['cmeans'] = k
    centroids = plot_clusters(ed_coords, k, 'cmeans')
    simulate(ed_coords, centroids, 'cmeans')
    
    plot_copex(ks)
    plot_dists(ks)
    plot_metrics(ks)

def test_models(X, ks):
    for model, k in ks.items():
        centroids, _, _ = plot_clusters(X, k, model)
        simulate(X, centroids, model)
    
    plot_copex(ks)
    plot_dists(ks)
    plot_metrics(ks)

if __name__ == '__main__':
    ed_coords = generate_ed_coords()
    #ks, _ = opt_models(ed_coords)
    ks = {
        'kmeans': 17,
        'kmedoids': 16,
        'cmeans': 16,
        'gk': 16,
    }
    test_models(ed_coords, ks)

####################