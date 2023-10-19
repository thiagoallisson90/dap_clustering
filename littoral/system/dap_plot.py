import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics import pairwise_distances
import pandas as pd

from littoral.system.dap_clustering import run_clustering
from littoral.system.dap_vars import img_dir, data_dir, model_names, base_dir
from littoral.system.dap_utils import capex_opex_calc, write_coords, define_colors

######################
# Plotting Functions #
######################

def plot_clusters(X, cntr, labels, cluster_points, model_name, folder_name):
    k = len(cntr)
    colors = plt.cm.rainbow(np.linspace(0, 1, k))

    plt.figure(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', alpha=0.7, s=100)

    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(model_name, fontsize=16)

    image_path = f'{base_dir}/antenna.jpg'
    image = plt.imread(image_path)
    for i, centroid in enumerate(cntr):
        imagebox = OffsetImage(image, zoom=0.1)
        ab = AnnotationBbox(imagebox, centroid, frameon=False)
        plt.gca().add_artist(ab)
        plt.text(centroid[0] + 1, centroid[1], f'     {i+1}', fontsize=16, fontweight="bold")

    plt.savefig(f'{img_dir}/{folder_name}/{k}gw.png')
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
    plt.title(f'Number of SMs in each Cluster {model_name}', fontsize=16)
    plt.savefig(f'{img_dir}/{folder_name}/{k}gw_chart.png', bbox_inches='tight')
    plt.clf()

    intra_cluster_distances = {}
    for label, points in enumerate(cluster_points):
        distances = pairwise_distances(points, metric='euclidean', n_jobs=-1)
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

    write_coords(cntr, f'data/{folder_name}/{k}gw_centroids.csv')
    df = pd.DataFrame(data)
    df.to_csv(f'{data_dir}/{folder_name}/{k}gw_distances.csv', index=False)

    return df

def plot_copex(ks, labels, capex_text='CapEx', opex_text='OpEx'):
    capex_values, opex_values = [], []
    for k in ks:
        data = capex_opex_calc(k)
        capex_values.append(data[0])
        opex_values.append(data[1])
            
    colors = define_colors(len(ks))

    plt.figure(figsize=(12, 8))
    plt.bar(labels, capex_values, color=colors)
    plt.xlabel('Clustering Models', fontsize=14)
    plt.ylabel('CapEx (K€)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(capex_text, fontsize=16)

    for i, value in enumerate(capex_values):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    plt.savefig(f'{img_dir}/capex_models.png')
    plt.clf()

    plt.figure(figsize=(12, 8))
    plt.bar(labels, opex_values, color=colors)
    plt.xlabel('Clustering Models', fontsize=14)
    plt.ylabel('OpEx (K€)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(opex_text, fontsize=16)
    
    for i, value in enumerate(opex_values):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.savefig(f'{img_dir}/opex_models.png')
    plt.clf()

def plot_dists(ks, labels, mean_dist, max_dist):
    width = 0.35
    x = np.arange(len(ks))

    plt.figure(figsize=(12, 8))
    plt.bar(x, mean_dist, width, label='Mean Distance', color='blue', hatch='++')
    plt.bar(x + width, max_dist, width, label='Maximium Distance', color='green', hatch='x')

    plt.xlabel('Clustering Algorithms', fontsize=14)
    plt.ylabel('Distances (m)', fontsize=14)
    plt.title('Intra-cluster Distances', fontsize=16)
    plt.xticks(x + width / 2, labels, fontsize=12)
    plt.yticks(fontsize=12)

    for i, (mean_value, max_value) in enumerate(zip(mean_dist, max_dist)):
        plt.text(x[i], mean_value, f'{mean_value:.2f}', ha='center', va='bottom', fontsize=12)
        plt.text(x[i] + width, max_value, f'{max_value:.2f}', ha='center', va='bottom', fontsize=12)

    plt.legend()

    plt.savefig(f'{img_dir}/dists_models.png')
    plt.clf()

"""
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
"""

######################