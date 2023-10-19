####################
# Global Variables #
####################

scratch_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch'
base_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering'
img_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering/imgs'
data_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering/data'
ed_pos_file = 'ed_pos_file.csv'
ed_out_file = 'ed_out_file.csv'
gw_pos_file = 'gw_pos_file.csv'
ns3_cmd = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/./ns3'
clustering_models = ['kmeans', 'kmedoids', 'cmeans', 'gk', 'rand', 'tests']
model_names = {
  'kmeans': 'K-Means',
  'kmedoids': 'K-Medoids',
  'cmeans':  'Fuzzy C-Means',
  'gk': 'Gustafson-Kessel',
  'rand': 'Rand'
}

####################