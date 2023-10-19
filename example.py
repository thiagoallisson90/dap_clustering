from littoral.system.dap_utils import generate_ed_coords
from littoral.system.dap_factory import KMeansFactory, KMedoidsFactory, CMeansSKFactory
from littoral.system.dap_factory import GKFactory, RandFactory
from littoral.system.dap_plot import plot_clusters, plot_copex, plot_dists
from littoral.system.dap_simulate import simulate

if __name__ == '__main__':
  coords = generate_ed_coords()
  

  ks = [17, 16, 16, 16, 16, 25]

  clfs = [KMeansFactory(ks[0]), KMedoidsFactory(ks[1]), CMeansSKFactory(ks[2]), GKFactory(ks[3]), 
          RandFactory(ks[4]), RandFactory(ks[5])]
  folders = ['kmeans', 'kmedoids', 'cmeans', 'gk', 'rand', 'rand']
  names = []
  text = ''
  mean_dist, max_dist = [], []

  for i in range(len(clfs)):
    clfs[i].fit(coords)
    name = clfs[i].name
    names.append(name)
    df = plot_clusters(coords, clfs[i].cluster_centers, clfs[i].labels, 
                  clfs[i].cluster_points(coords), name, folders[i])
    mean_dist.append(df['mean_dist'].mean())    
    max_dist.append(df['max_dist'].mean())

    if(i == 0):
        text = name
    elif(i < len(clfs) - 1):
        text = text + ', ' + name
    else:
        text = text + ' and ' + name

  plot_copex(ks, labels=names, capex_text=f'CapEx - {text}', opex_text=f'OpEx - {text}')
  plot_dists(ks, labels=names, mean_dist=mean_dist, max_dist=max_dist)

  df_sims = []
  load = 5
  for i in range(len(clfs)):
    print(simulate(coords, clfs[i].cluster_centers, folders[i], load=load))
