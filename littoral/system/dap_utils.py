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

#####################