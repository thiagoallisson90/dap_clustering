import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.cluster import KElbowVisualizer
from gapstatistic import optimalK, optimalC

scratch_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch'
dap = 'dap_clustering'
base_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering'
img_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering/imgs'
data_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering/data'
ed_pos_file = 'ed_pos_file.csv'
ed_out_file = 'ed_out_file.csv'
gw_pos_file = 'gw_pos_file.csv'
ns3_cmd = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/./ns3'

def generate_ed_coords(n_points=2000, axis_range = 10000):
  np.random.seed(42)
  x = np.random.uniform(0, axis_range, n_points)
  y = np.random.uniform(0, axis_range, n_points)
  
  return np.column_stack((x, y))

def write_coords(data, filename):
    with open(f'{base_dir}/{filename}', mode='w') as file:
        for d in data:
            file.write(f'{d[0]},{d[1]}\n')

def simulate(coords, centroids, model, ed_pos_file=ed_pos_file, ed_out_file=ed_out_file, gw_pos_file=gw_pos_file):
    script='scratch/dap_clustering.cc'
    n_gw = len(centroids)
    n_simulatons = 30
    
    write_coords(coords, ed_pos_file)
    write_coords(centroids, gw_pos_file)
    for i in range(1, n_simulatons+1):
        run_cmd = \
            f'{ns3_cmd} run "{script} --edPos={ed_pos_file} --edOutputFile={ed_out_file} --gwPos={gw_pos_file} --nGateways={n_gw} --nRun={i} --cModel={model}"'
        os.system(run_cmd)

def write_scores(data, filename):
    with open(f'{base_dir}/{filename}', mode='a') as file:
        file.write(f'{data["k"]},{data["score"]}\n')

def opt_kmodel(data, model_name='kmeans', metric_name='elbow'):
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
        'gap': 'gap_statistic'
    }

    fn_obj = lambda oldScore, newScore: oldScore > newScore if metric_name == 'elbow' \
        else lambda oldScore, newScore: oldScore < newScore


    print(f'Running ({model_name},{metric_name})')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    min_clusters = 1 if metric_name in ['elbow', 'gap'] else 2
    max_clusters = 30
    num_iter = 30
    n_refs = 500 if model_name == 'kmeans' else 50

    opt_dir = f'data/{model_name}/{metric_name}'
    best_k = -1
    best_score = -1
    plot_score = None

    if not metric_name == 'gap':
        for i in range(num_iter): 
            visualizer = KElbowVisualizer(models[model_name], metric=metrics[metric_name],
                                          k=(min_clusters, max_clusters+1), timings=False)    
            visualizer.fit(data)
            
            k = visualizer.elbow_value_
            score = visualizer.elbow_score_
            if(i == 0 or fn_obj(best_score, score)):
                best_k = k
                best_score = score
                plot_score = visualizer.k_scores_

            write_scores({'k': k, 'score': score}, f'{opt_dir}/daps.csv')
            
            print(f'k defined in iteration {i+1} with {metric_name} method = {k}')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    else:
        for i in range(num_iter-10):
            k, gap_history = optimalK(data, nrefs=n_refs, maxClusters=max_clusters+1, model_name=model_name)            
            score = gap_history['gap'][k-1]
            
            if(i == 0 or best_score < score):
                best_k = k
                best_score = score
                plot_score = gap_history['gap']

            write_scores({'k': k, 'score': score}, f'{opt_dir}/daps.csv')
            
            print(f'k defined in iteration {i+1} with {metric_name} = {k}')
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')    

    print(f'{best_k} DAPs have score equals to {best_score}')

    plt.clf()

    plt.plot(range(min_clusters, max_clusters+1, 1), plot_score)
    plt.title(f'DAP optimal quantity - {model_name} and {metric_name} method (k={best_k})', fontsize=13)
    plt.xlabel('k', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(f'{img_dir}/{model_name}/{metric_name}/{best_k}_daps.png')    
    plt.clf()
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

def opt_kmodels():
    models_name = ['kmeans', 'kmedoids']
    metric_names = ['elbow', 'silhouette', 'calinski', 'gap']    

    ed_coords = generate_ed_coords()
    write_coords(ed_coords, ed_pos_file)

    ks = dict()

    for model in models_name:
        for metric in metric_names:
            opt_kmodel(ed_coords, model, metric)

def capex_opex_calc(n_gws):
    CBs, Cins, Cset, Txinst = 1, 2, 0.1, 4

    Cman = 12.5/100
    t=1

    capex = n_gws *(CBs + Cins + Cset + Txinst)
    opex = (Cman*capex + n_gws)*t

    return capex, opex

def plot_copex(ks):
    n_gws_kmeans, n_gws_kmedoids, n_gws_cmeans = ks['kmeans'], ks['kmedoids'], ks['cmeans']
    capex1, opex1 = capex_opex_calc(n_gws_kmeans)
    capex2, opex2 = capex_opex_calc(n_gws_kmedoids)
    capex3, opex3 = capex_opex_calc(n_gws_cmeans)

    capex_values = [capex1, capex2, capex3]
    opex_values = [opex1, opex2, opex3]

    scenario_labels = ['K-Means', 'K-Medoids', 'C-Means']

    plt.figure(figsize=(12, 8))
    plt.bar(scenario_labels, capex_values, color=['red', 'green', 'blue'])
    plt.xlabel('Clustering Models', fontsize=14)
    plt.ylabel('CapEx (K€)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('CapEx - K-Means, K-Medoids and C-Means', fontsize=16)

    for i, value in enumerate(capex_values):
        plt.text(i, value, str(value), ha='center', va='bottom', fontsize=12)

    plt.savefig(f'{img_dir}/capex_kmodels.png')
    plt.clf()

    plt.figure(figsize=(12, 8))
    plt.bar(scenario_labels, opex_values, color=['red', 'green', 'blue'])
    plt.xlabel('Clustering Models', fontsize=14)
    plt.ylabel('OpEx (K€)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('OpEx - K-Means, K-Medoids and C-Means', fontsize=16)
    
    for i, value in enumerate(opex_values):
        plt.text(i, value, str(value), ha='center', va='bottom', fontsize=12)
    
    plt.savefig(f'{img_dir}/opex_kmodels.png')
    plt.clf()

def plot_kclusters(X, model, name='kmeans'):
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from sklearn.metrics import pairwise_distances

    kmeans_labels = model.labels_
    centroids = model.cluster_centers_
    title = {
        'kmeans': 'KMeans Clusters',
        'kmedoids': 'KMedoids Clusters'
    }

    n_clusters = len(centroids)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='rainbow', alpha=0.7, s=100)

    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title[name], fontsize=16)

    image_path = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering/imgs/antenna.jpg'
    image = plt.imread(image_path)
    for i, centroid in enumerate(centroids):
        imagebox = OffsetImage(image, zoom=0.1)
        ab = AnnotationBbox(imagebox, centroid, frameon=False)
        plt.gca().add_artist(ab)
        plt.text(centroid[0] + 1, centroid[1], f'     {i+1}', fontsize=16, fontweight="bold")

    plt.savefig(f'{img_dir}/{name}/{len(centroids)}gw.png')
    plt.clf()

    cluster_counts = np.bincount(kmeans_labels)

    sorted_indices = np.argsort(cluster_counts)[::-1]
    sorted_cluster_counts = cluster_counts[sorted_indices]
    sorted_cluster_labels = [f'Cluster {i+1}' for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    
    plt.barh(range(n_clusters), sorted_cluster_counts, tick_label=sorted_cluster_labels, color=colors)
    plt.xlabel('SMs per Cluster', fontsize=14)
    plt.ylabel('Clusters', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title('Number of SMs in each Cluster', fontsize=16)
    plt.savefig(f'{img_dir}/{name}/{len(centroids)}gw_chart.png', bbox_inches='tight')
    plt.clf()

    cluster_points = {}
    for i in range(n_clusters):
        cluster_points[i] = []

    for i, label in enumerate(kmeans_labels):
        cluster_points[label].append(X[i])

    intra_cluster_distances = {}
    for label, points in cluster_points.items():
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
        'cluster': list(range(1, n_clusters+1)),
        'mean_dist': [round(d, 2) for d in average_intra_cluster_distances.values()],
        'max_dist': [round(d, 2) for d in max_intra_cluster_distances.values()]
    }

    write_coords(centroids, f'data/{name}/{len(centroids)}gw_centroids.csv')
    pd.DataFrame(data).to_csv(f'{data_dir}/{name}/{len(centroids)}gw_distances.csv', index=False)

def plot_cclusters(X, k):
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from sklearn.metrics import pairwise_distances
    from skfuzzy.cluster import cmeans

    cntr, u, u0, d, jm, p, fpc = cmeans(X.T, c=k, m=2, error=0.005, maxiter=1000)
    n_clusters = len(cntr)

    labels = np.argmax(u, axis=0)
    
    cluster_points = {}
    for i in range(n_clusters):
        cluster_points[i] = []

    for i, label in enumerate(labels):
        cluster_points[label].append(X[i])

    colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))

    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', alpha=0.7, s=100)

    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Fuzzy C-Means', fontsize=16)

    image_path = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering/imgs/antenna.jpg'
    image = plt.imread(image_path)
    for i, centroid in enumerate(cntr):
        imagebox = OffsetImage(image, zoom=0.1)
        ab = AnnotationBbox(imagebox, centroid, frameon=False)
        plt.gca().add_artist(ab)
        plt.text(centroid[0] + 1, centroid[1], f'     {i+1}', fontsize=16, fontweight="bold")

    plt.savefig(f'{img_dir}/cmeans/{n_clusters}gw.png')
    plt.clf()

    cluster_counts = np.bincount(labels)

    sorted_indices = np.argsort(cluster_counts)[::-1]
    sorted_cluster_counts = cluster_counts[sorted_indices]
    sorted_cluster_labels = [f'Cluster {i+1}' for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    
    plt.barh(range(n_clusters), sorted_cluster_counts, tick_label=sorted_cluster_labels, color=colors)
    plt.xlabel('SMs per Cluster', fontsize=14)
    plt.ylabel('Clusters', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title('Number of SMs in each Cluster', fontsize=16)
    plt.savefig(f'{img_dir}/cmeans/{n_clusters}gw_chart.png', bbox_inches='tight')
    plt.clf()

    intra_cluster_distances = {}
    for label, points in cluster_points.items():
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
        'cluster': list(range(1, n_clusters+1)),
        'mean_dist': [round(d, 2) for d in average_intra_cluster_distances.values()],
        'max_dist': [round(d, 2) for d in max_intra_cluster_distances.values()]
    }

    write_coords(cntr, f'data/cmeans/{n_clusters}gw_centroids.csv')
    pd.DataFrame(data).to_csv(f'{data_dir}/cmeans/{n_clusters}gw_distances.csv', index=False)

    return cntr

def plot_metrics(ks):
    import seaborn as sns

    n_gws_kmeans, n_gws_kmedoids, n_gws_cmeans = ks['kmeans'], ks['kmedoids'], ks['cmeans']

    names=['sent', 'received', 'ul-pdr', 'rssi', 'snr', 'delay']
    df_partial_kmeans = \
        pd.read_csv(f'{data_dir}/kmeans/tracker_1_unconfirmed_buildings{n_gws_kmeans}gw.csv', names=names)

    df_partial_kmedoids = \
        pd.read_csv(f'{data_dir}/kmedoids/tracker_1_unconfirmed_buildings{n_gws_kmedoids}gw.csv', names=names)
    
    df_partial_cmeans = \
        pd.read_csv(f'{data_dir}/cmeans/tracker_1_unconfirmed_buildings{n_gws_cmeans}gw.csv', names=names)
    
    labels = ['K-Means', 'K-Medoids', 'C-Means']
    metrics = ['ul-pdr', 'rssi', 'snr', 'delay']
    units = {
        'ul-pdr': '',
        'rssi': '',
        'snr': ' (dB)',
        'delay': ' (s)',
    }
    for metric in metrics:
        datas = {
            'K-Means': df_partial_kmeans[metric], 
            'K-Medoids': df_partial_kmedoids[metric],
            'C-Means': df_partial_cmeans[metric]
        }
        upper_metric = metric.upper()

        plt.figure(figsize=(12, 8))
        sns.boxplot(pd.DataFrame(datas), width=0.5)
        plt.xlabel('Clustering Models', fontsize=14)
        plt.ylabel(upper_metric + units[metric], fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.title(f'{upper_metric} - K-Means, K-Medoids and C-Means', fontsize=16)
        plt.legend(labels=labels, title='Clustering Models', fontsize=10)
        
        plt.savefig(f'{img_dir}/{metric}_kmeans_{n_gws_kmeans}_kmedoids_{n_gws_kmedoids}_cmeans{n_gws_cmeans}.png')
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

    datas = {
        'K-Means': [(init_energy - df['remainder_energy'].mean()) for df in kmeans_energy],
        'K-Medoids': [(init_energy - df['remainder_energy'].mean()) for df in kmedoids_energy],
        'C-Means': [(init_energy - df['remainder_energy'].mean()) for df in cmeans_energy]
    }

    plt.figure(figsize=(12, 8))
    sns.boxplot(pd.DataFrame(datas), width=0.5)
    plt.xlabel('Clustering Models', fontsize=14)
    plt.ylabel('Consumed Energy (J)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Consumed Energy - KMeans, KMedoids and C-Means', fontsize=16)
    plt.legend(labels=labels, title='Clustering Models', fontsize=10)
    
    plt.savefig(f'{img_dir}/energy_kmeans_{n_gws_kmeans}_kmedoids_{n_gws_kmedoids}.png')
    plt.clf()

def simulate_models():
    from sklearn.cluster import KMeans
    from sklearn_extra.cluster import KMedoids

    ed_coords = generate_ed_coords()

    model_names = ['kmeans', 'kmedoids']
    models = {
        'kmeans': lambda k: KMeans(k, n_init='auto'),
        'kmedoids': lambda k: KMedoids(k),
    }
    metric_names = ['gap']
    ks = {}

    for model in model_names:
        for metric in metric_names:
            df = pd.read_csv(f'{base_dir}/data/{model}/{metric}/daps.csv', names=['k', 'gap'])
            k = df.loc[df['gap'].idxmax(), 'k']
            ks[model] = k
            clf = models[model](k)
            clf.fit(ed_coords)
            centroids = clf.cluster_centers_
            plot_kclusters(ed_coords, clf, model)
            #simulate(ed_coords, centroids, model)

    # Simulating C-Means
    df = pd.read_csv(f'{data_dir}/cmeans/gap/daps.csv').drop(0, axis=0)
    k = df.loc[df['gap'].idxmax(), 'clusterCount']
    ks['cmeans'] = k
    centroids = plot_cclusters(ed_coords, k)
    #simulate(ed_coords, centroids, 'cmeans')
    
    plot_copex(ks)
    plot_metrics(ks)

def opt_cmeans():
    ed_coords = generate_ed_coords() 

    max_clusters = 30

    k, results, _ = optimalC(ed_coords.T, nrefs=5, maxClusters=max_clusters+1)

    plt.plot(range(1, max_clusters+1, 1), results['gap'])
    plt.title(f'DAP optimal quantity - C-Means and Gap Statistics (k={k})', fontsize=14)
    plt.grid(True)
    plt.xlabel('k', fontsize=12)
    plt.ylabel('Gaps', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig(f'{img_dir}/cmeans/gap/{k}_daps.png')
    plt.clf()

    results.to_csv(f'{data_dir}/cmeans/gap/daps.csv', index=False)
    print(f'DAP optimal quantity - C-Means and Gap Statistics (k={k})')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

if __name__ == '__main__':
    # opt_kmodels()
    # opt_cmeans()
    simulate_models()
