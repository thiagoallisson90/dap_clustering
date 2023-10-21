import pandas as pd
from littoral.system.dap_vars import data_dir

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
