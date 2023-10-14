import numpy as np
from sklearn_extra.cluster import KMedoids
from gapstatistic import optimalK
import matplotlib.pyplot as plt

def gen_coords(axis_range=10000, n_points=2000):
  np.random.seed(42)
  x = np.random.uniform(0, axis_range, n_points)
  y = np.random.uniform(0, axis_range, n_points)

  np.random.seed(None)

  return np.column_stack((x, y))

def optmizeK():
  print(KMedoids.__name__)
  k, history = optimalK(gen_coords(), KMedoids, max_clusters=25)
  print(f'GW optimal number is {k}')
  plt.plot(range(1, 25), history['gap'])
  plt.grid(True)
  plt.xlabel('k', fontsize=14)
  plt.ylabel('Gap', fontsize=14)
  plt.xticks(range(1, 25, 1), fontsize=10)
  plt.yticks(fontsize=10)
  plt.show()

if __name__ == '__main__':
  from skcmeans.algorithms import GustafsonKesselMixin, Probabilistic

  class ProbabilisticGustafsonKessel(GustafsonKesselMixin, Probabilistic):
    pass

  pgk = ProbabilisticGustafsonKessel(10)
  pgk.fit(gen_coords())

  print(pgk.centers)

