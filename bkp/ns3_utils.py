import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

scratch_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch'
dap = 'dap_clustering'
base_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering'
img_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering/imgs'
data_dir = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/scratch/dap_clustering/data'
ed_pos_file = 'ed_pos_file.csv'
ed_out_file = 'ed_out_file.csv'
gw_pos_file = 'gw_pos_file.csv'
ns3_cmd = '/home/thiago/Documentos/Doutorado/Simuladores/ns-3-dev/./ns3'

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
    n_simulatons = 1
    
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

def test_coverage():
    import seaborn as sns

    df_ok = None
    radius_ok = None

    for radius in [2709.071, 2709.072, 2709.073, 2709.074]:

        simulate([[radius, radius]], [[0, 0]], model='tests', radius=radius)

        df = pd.read_csv(f'{data_dir}/tests/tracker_1_unconfirmed_buildings1gw.csv', 
                        names=['sent', 'rec', 'pdr', 'rssi', 'snr', 'delay', 'radius'])
        row = df.loc[df['rec'] == 0.0]
        if(len(row) > 0):
            break
        else:
            radius_ok = radius
            df_ok = df
    
        os.system('clear')
    os.system('clear')

    key1 = str(radius_ok)

    print('Maximium radius = ', radius_ok)
    print(f'Minimum RSSI = {df_ok["rssi"].min()} (dBm)')

    plt.figure(figsize=(12, 8))
    
    ax = sns.boxplot(pd.DataFrame({key1: df_ok['rssi']}))
    ax.set_ylim(-130.0, -143.0)

    plt.xlabel(f'Coverage Radius (m)', fontsize=14)
    plt.ylabel('RSSI (dBm)', fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title(f'Maximium Coverage Radius Analysis', fontsize=16)
    plt.savefig(f'{img_dir}/tests/max_radius.png')
    #plt.show()
    plt.clf()

"""
if __name__ == '__main__':
    test_coverage()
"""