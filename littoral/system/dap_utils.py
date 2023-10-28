import numpy as np
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
            [pd.read_csv(f'{data_dir}/{folder}/nRun_{i}_{k}gws_battery-level.txt', names=col_names) \
                for i in range(1, n_reps+1)]
        
        energy_values[j] = [initial_energy - df['remainder_energy'].mean() for df in dfs]

    return energy_values

def normal_test(data):
    alpha = 0.05
    p = 0.0
    size = len(data)

    if(size > 4 and size <= 30):
        from scipy.stats import shapiro        
        _, p = shapiro(data)
    elif(size > 30 and size <= 50):
        from statsmodels.stats.diagnostic import lilliefors
        _, p = lilliefors(data, dist='norm', pvalmethod='approx')
    
    if(p >= alpha):
        return True, p
    
    return False, p

def test_t(data, pop_mean):
    from scipy.stats import ttest_1samp

    alpha = 0.05
    _, p = ttest_1samp(data, popmean=pop_mean)

    return (p >= alpha), p

def compute_sf(k, folder, n_sims=30):
    df = pd.read_csv(f'{data_dir}/{folder}/{k}gw_sf.csv', names=['ED', 'GW', 'RX', 'SF'])
    return np.round(df['SF'].value_counts() / n_sims)

def run_sf_and_tests(df_sims, ks, folders, energy_values, names):
    from littoral.system.dap_plot import plot_sf

    sample_size = 5
    pop_size = df_sims[0].shape[0]
    columns = ['ul-pdr', 'rssi', 'snr', 'delay']

    for i in range(len(ks)):
        k = ks[i]
        folder = folders[i]

        plot_sf(compute_sf(k, folder), k, folder)

        df_final = df_sims[i][columns].copy()
        df_final['energy'] = energy_values[i]

        print(f'{names[i]} Analysis:')

        ps = {}
        means = {}

        for col in df_final.columns:
            result = normal_test(df_final[col])
            ps[col] = [result[1]]
            if(result[0]):
                print(f'{col} sample is normal, with p-value = {ps[col]}')
            else:
                print(f'{col} sample isn\'t normal, with p-value = {ps[col]}')
        pd.DataFrame(ps).to_csv(f'{data_dir}/{folder}/normal_tests_{k}gws.csv', index=False)

        for col in df_final.columns:
            result = test_t(df_final[col][0:sample_size], df_final[col].mean())
            means[col] = [result[1]]
            if(result[0]):
                print(f'Sample mean for {col} is equal population mean ({pop_size}), with p = {means[col]}')
            else:
                print(f'Sample mean for {col} isn\'t equal population mean ({pop_size}), with p = {means[col]}')
        pd.DataFrame(means).to_csv(f'{data_dir}/{folder}/mean_tests_{k}gws.csv', index=False)

#####################