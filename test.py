from littoral.system.dap_utils import generate_ed_coords
from littoral.system.dap_factory import KMeansFactory, KMedoidsFactory, CMeansSKFactory, GKFactory, RandFactory
from littoral.system.dap_plot import plot_clusters, plot_copex, plot_dists

if __name__ == '__main__':
  coords = generate_ed_coords()

  ks = [17, 16, 16, 16, 16, 25]

  clf1 = KMeansFactory(ks[0])
  clf1.fit(coords)

  clf2 = KMedoidsFactory(ks[1])
  clf2.fit(coords)

  clf3 = CMeansSKFactory(ks[2])
  clf3.fit(coords)

  clf4 = GKFactory(ks[3])
  clf4.fit(coords)

  clf5 = RandFactory(ks[4])
  clf5.fit(coords)

  clf6 = RandFactory(ks[5])
  clf6.fit(coords)

  clfs = [clf1, clf2, clf3, clf4, clf5, clf6]
  folders = ['kmeans', 'kmedoids', 'cmeans', 'gk', 'rand', 'rand']
  names = []
  text = ''
  mean_dist, max_dist = [], []

  for i in range(len(clfs)):
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
        text = text + ' and' + name

  plot_copex(ks, labels=names, capex_text=f'CapEx - {text}', opex_text=f'OpEx - {text}')
  plot_dists(ks, labels=names, mean_dist=mean_dist, max_dist=max_dist)