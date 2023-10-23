import os
import pandas as pd

from littoral.system.dap_utils import write_coords
from littoral.system.dap_vars import ed_pos_file, ed_out_file, gw_pos_file
from littoral.system.dap_vars import data_dir, ns3_cmd

def simulate(coords, centroids, folder, ed_pos_file=ed_pos_file, ed_out_file=ed_out_file, 
             gw_pos_file=gw_pos_file, radius=10000, load=5):
    script='scratch/dap_clustering.cc'
    n_gw = len(centroids)
    n_simulatons = 50

    write_coords(coords, ed_pos_file)
    write_coords(centroids, gw_pos_file)
    filename = f'{data_dir}/{folder}/tracker_unconfirmed_buildings{n_gw}gw.csv'
    with open(filename, mode="w") as file:
        file.write('')
        file.close()

    params01 = f'--edPos={ed_pos_file} --edOutputFile={ed_out_file} --gwPos={gw_pos_file} --nGateways={n_gw}'
    params02 = f'--cModel={folder} --radius={radius} --nDevices={len(coords)} --lambda={load}'

    for i in range(1, n_simulatons+1):
        run_cmd = \
            f'{ns3_cmd} run "{script} {params01} {params02} --nRun={i}"'
        os.system(run_cmd)
    
    col_names = ['sent', 'received', 'ul-pdr', 'rssi', 'snr', 'delay']
    return pd.read_csv(filename, names=col_names)
