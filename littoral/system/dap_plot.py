import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics import pairwise_distances
import pandas as pd
import seaborn as sns

from littoral.system.dap_vars import img_dir, data_dir, base_dir
from littoral.system.dap_utils import write_coords, define_colors

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

def plot_capex_opex(capex, opex, labels):           
    colors = define_colors(len(capex))

    plt.figure(figsize=(12, 8))
    plt.bar(labels, capex, color=colors)
    plt.xlabel('Clustering Models', fontsize=14)
    plt.ylabel('CapEx (K€)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('CapEx of the Clustering Algorithms', fontsize=16)

    for i, value in enumerate(capex):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12)

    plt.savefig(f'{img_dir}/capex_models.png')
    plt.clf()

    plt.figure(figsize=(12, 8))
    plt.bar(labels, opex, color=colors)
    plt.xlabel('Clustering Models', fontsize=14)
    plt.ylabel('OpEx (K€)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('OpEx of the Clustering Algorithms', fontsize=16)
    
    for i, value in enumerate(opex):
        plt.text(i, value, f'{value:.2f}', ha='center', va='bottom', fontsize=12)
    
    plt.savefig(f'{img_dir}/opex_models.png')
    plt.clf()

def plot_dists(mean_dist, max_dist, labels):
    width = 0.35
    x = np.arange(len(mean_dist))

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

def plot_metric(datas, labels, title_text, y_text, metric_name):
    plt.figure(figsize=(12, 8))    
    sns.boxplot(datas, width=0.5, palette=define_colors(len(labels)))
    plt.xlabel('Clustering Models', fontsize=14)
    plt.ylabel(y_text, fontsize=14)
    plt.xticks(range(len(labels)), labels, fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(title_text, fontsize=16)
    plt.legend(labels=labels, title='Clustering Models', fontsize=10)   
    plt.savefig(f'{img_dir}/{metric_name}_models.png')
    plt.clf()

metric_names = ['delay', 'energy', 'rssi', 'snr', 'ul-pdr']

def plot_delay(delay_df, labels):
    name = 'Delay'
    title = f'{name} of the Clustering Algorithms'
    unit = 's'
    y = f'{name} ({unit})'
    plot_metric(delay_df, labels, title, y, name)

def plot_energy(energy_df, labels):
    name = 'Consumed Energy'
    title = f'{name} by the Clustering Algorithms'
    unit = 'J'
    y = f'{name} ({unit})'
    plot_metric(energy_df, labels, title, y, name)

def plot_rssi(rssi_df, labels):
    name = 'RSSI'
    title = f'{name} of the Clustering Algorithms'
    unit = 'dBm'
    y = f'{name} ({unit})'
    plot_metric(rssi_df, labels, title, y, name)

def plot_snr(snr_df, labels):
    name = 'SNR'
    title = f'{name} of the Clustering Algorithms'
    unit = 'dB'
    y = f'{name} ({unit})'
    plot_metric(snr_df, labels, title, y, name)

def plot_ulpdr(ulpdr_df, labels):
    name = 'UL-PDR'
    title = f'{name} of the Clustering Algorithms'
    unit = '%'
    y = f'{name} ({unit})'
    plot_metric(ulpdr_df, labels, title, y, name)

plot_metrics = {
    metric_names[0]: plot_delay,
    metric_names[1]: plot_energy,
    metric_names[2]: plot_rssi,
    metric_names[3]: plot_snr,
    metric_names[4]: plot_ulpdr,
}

######################