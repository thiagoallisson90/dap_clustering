from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import pandas as pd

from littoral.system.dap_utils import generate_ed_coords, capex_opex_calc
from littoral.system.dap_factory import KMeansFactory, KMedoidsFactory, CMeansFactory
from littoral.system.dap_factory import GKFactory, RandFactory
from littoral.system.dap_plot import plot_clusters, plot_capex_opex, plot_dists, plot_wcss
from littoral.system.dap_plot import metric_names, plot_metrics, plot_sf
from littoral.system.dap_simulate import simulate
from littoral.system.dap_elbow import CrispElbow, CMeansElbow, GKElbow
from littoral.system.dap_utils import compute_consumed_energy, normal_test, test_t
from littoral.system.dap_utils import run_sf_and_tests
from littoral.system.dap_vars import data_dir

if __name__ == '__main__':
  coords = generate_ed_coords()

  #ks_range = range(15, 30)
  ks = []
  metric = 'calinski_harabasz'

  print('#########################################################################################')  
  print('Computing the optimal values to k')
  ks = [16, 18, 16, 15] # 16, 18, 16, 15
  """best_k = CrispElbow().execute(coords, ks=ks_range, model=KMeans(n_init='auto', init='k-means++'), metric=metric)
  print(f'Best k to K-Means: {best_k}')
  ks.append(best_k[0])
  
  best_k = CrispElbow().execute(coords, ks=ks_range, model=KMedoids(init='k-medoids++'), metric=metric)
  print(f'Best k to K-Medoids: {best_k}')
  ks.append(best_k[0])

  best_k = CMeansElbow().execute(coords, ks=ks_range, metric=metric)
  print(f'Best k to Fuzzy C-Means: {best_k}')
  ks.append(best_k[0])
  
  best_k = GKElbow().execute(coords, ks=ks_range, metric=metric)
  print(f'Best k to Gustafson-Kessel: {best_k}')
  ks.append(best_k[0])"""

  ks.append(16) # Rand16
  ks.append(25) # Rand25
  
  print('#########################################################################################')  
  print('Plotting CapEx, OpEx, Intra-cluster Distances and WCSS')

  clfs = [KMeansFactory(ks[0]), KMedoidsFactory(ks[1]), CMeansFactory(ks[2]), GKFactory(ks[3]), 
          RandFactory(ks[4]), RandFactory(ks[5])]
  folders = ['kmeans', 'kmedoids', 'cmeans', 'gk', 'rand', 'rand']
  names, mean_dist, max_dist = [], [], []
  capex_values, opex_values = [], []
  wcss =  []

  for i in range(len(clfs)):
    clfs[i].fit(coords)
    name = clfs[i].name
    names.append(name)
    df = plot_clusters(coords, clfs[i].cluster_centers, clfs[i].labels, 
                  clfs[i].cluster_points(coords), name, folders[i])
    mean_dist.append(df['mean_dist'].max())    
    max_dist.append(df['max_dist'].max())

    k = len(clfs[i].cluster_centers)
    capex, opex = capex_opex_calc(k)
    capex_values.append(capex)
    opex_values.append(opex)
    wcss.append(clfs[i].wcss)

  plot_capex_opex(capex_values, opex_values, labels=names)
  plot_dists(mean_dist=mean_dist, max_dist=max_dist, labels=names)
  plot_wcss(wcss, labels=names)
  
  print('#########################################################################################')  
  print('Simulating and plotting metrics: Delay, Energy, RSSI, SNR and UL-PDR')
  #load = 1 / (5 * 60) # 1pkt/5min -> 1pkt/300s
  load = 1
  df_sims = \
    [simulate(coords, clfs[i].cluster_centers, folders[i], load=load, setUpSF=1, 
              connFile=f'{data_dir}/{folders[i]}/{ks[i]}gw_labels.csv') for i in range(len(clfs))]  

  for m in metric_names:
    if(m != 'energy'):
      plot_metrics[m]([df[m] for df in df_sims], names)

  energy_values = [compute_consumed_energy(ks[i], folder=folders[i]) for i in range(len(ks))]
  plot_metrics['energy'](energy_values, names)

  print('#########################################################################################')  
  print('Normal and T tests, as well as SF analysis')
  
  run_sf_and_tests(df_sims, ks, folders, energy_values, names)

  print('#########################################################################################')