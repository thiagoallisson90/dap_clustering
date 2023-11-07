import numpy as np
import os
import pandas as pd

from littoral.system.dap_utils import write_coords
from littoral.system.dap_vars import ed_pos_file, ed_out_file, gw_pos_file
from littoral.system.dap_vars import data_dir, ns3_cmd

def simulate(coords, centroids, folder, ed_pos_file=ed_pos_file, ed_out_file=ed_out_file, 
             gw_pos_file=gw_pos_file, radius=10000, load=1, setUpSF=0, connFile=''):
    script='scratch/dap_clustering/dap_clustering.cc'
    n_gw = len(centroids)
    n_simulatons = 30

    write_coords(coords, ed_pos_file)
    write_coords(centroids, gw_pos_file)
    
    filename = f'{data_dir}/{folder}/tracker_unconfirmed_buildings{n_gw}gw.csv'
    with open(filename, mode="w") as file:
        file.write('')
        file.close()
    
    sf_filename = f'{data_dir}/{folder}/{n_gw}gw_sf.csv'
    with open(sf_filename, mode="w") as file:
        file.write('')
        file.close()

    params01 = f'--edPos={ed_pos_file} --edOutputFile={ed_out_file} --gwPos={gw_pos_file} --nGateways={n_gw}'
    params02 = f'--cModel={folder} --radius={radius} --nDevices={len(coords)} --lambda={load}'
    params03 = f'--setUpSF={setUpSF} --connFile={connFile}'

    for i in range(1, n_simulatons+1):
        run_cmd = f'{ns3_cmd} run "{script} {params01} {params02} {params03} --nRun={i}"'
        os.system(run_cmd)
    
    col_names = ['sent', 'received', 'ul-pdr', 'rssi', 'snr', 'delay']
    return pd.read_csv(filename, names=col_names)

def simulate_tests(coords, centroids, cmodel, ed_pos_file=ed_pos_file, gw_pos_file=gw_pos_file, 
                   radius=10000):
    script = 'scratch/rssi.cc'
    n_gw = len(centroids)
    n_simulations = 30

    write_coords(coords, ed_pos_file)
    write_coords(centroids, gw_pos_file)
    
    filename = f'{data_dir}/{cmodel}/tracker_{n_gw}gw.csv'
    with open(filename, mode="w") as file:
        file.write('')
        file.close()

    params01 = f'--edPosA={ed_pos_file} --gwPosA={gw_pos_file} --nGWs={n_gw}'
    params02 = f'--cModel={cmodel} --size={radius} --nDevs={len(coords)}'

    for i in range(1, n_simulations+1):
        run_cmd = f'{ns3_cmd} run "{script} {params01} {params02} --numberRun={i}"'
        os.system(run_cmd)
    
    names = [f'rssi{i}' for i in range(n_gw)]
    df = pd.read_csv(f'{data_dir}/{cmodel}/tracker_{n_gw}gw.csv', names=names)
    n_rows = len(coords)
    result = np.zeros((n_rows, n_gw))

    for i in range(df.shape[0]):
        row = i % n_rows
        j = 0
        for col in df.columns:
            result[row][j] = result[row][j] + df[col][i]
            j = j + 1
    
    result = result / n_simulations
    return pd.DataFrame(result, columns=names)
