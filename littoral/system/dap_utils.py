import numpy as np
import os
import pandas as pd

from littoral.system.dap_vars import *

#####################
# Utility Functions #
#####################

def generate_ed_coords(n_points=2000, axis_range=10000, seed=42):
  np.random.seed(seed)
  coords = np.random.uniform(0, axis_range, (n_points, 2))
  np.random.seed(None)
  
  return coords

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

def capex_opex_calc(n_daps):
    CBs, Cins, Cset, Txinst = 1, 2, 0.1, 4

    Cman = 0.125
    # Clease, Celet, Ctrans, t = 1, 1, 0.1, 1

    capex = n_daps * (CBs + Cins + Cset + Txinst)
    #opex = (Cman*capex + n_gws * (Clease + Celet + Ctrans)) * t
    opex = (Cman * capex + n_daps)

    return capex, opex

def write_scores(data, filename):
    with open(f'{base_dir}/{filename}', mode='a') as file:
        file.write(f'{data["k"]},{data["score"]}\n')

def define_colors(k, seed=42):
    np.random.seed(seed)
    colors = \
        [(np.random.random(), np.random.random(), np.random.random()) for _ in range(k)]
    np.random.seed(None)
    return colors

def compute_consumed_energy(ks, folders, data_dir=data_dir, n_reps=30):
    initial_energy = 10000 # in J
    col_names = ['uid', 'remainder_energy']

    energy_values = [0 for _ in ks]
    for j in range(len(ks)):
        k = ks[j]
        folder = folders[j]
        dfs = \
            [pd.read_csv(f'{data_dir}/{folder}/nRun_{i}_{k}gws_battery-level.txt', names=col_names).drop(0, axis=0) \
                for i in range(1, n_reps+1)]
        
        energy_values[j] = [initial_energy - df['remainder_energy'].mean() for df in dfs]

    return energy_values

#####################