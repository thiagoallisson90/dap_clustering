import numpy as np
import pandas as pd
from pymoo.core.problem import Problem
from ns3_utils import generate_ed_coords, simulate, data_dir

class DapPlacement(Problem):

    def __init__(self, n_var, n_obj, xl=10000.0, xu=10000.0):
        super().__init__(n_var=n_var,
                         n_obj=n_obj,
                         xl=xl,
                         xu=xu)
        self.n_daps = n_var / 2
        self.ed_coords = generate_ed_coords()


class DapPlacement10(DapPlacement):

    def __init__(self):
        super().__init__(n_var=20,
                         n_obj=2)

    def _evaluate(self, centroids, out, *args, **kwargs):
        simulate(self.ed_coords, centroids, 'tests')
        df = pd.read_csv(f'{data_dir}/tests/tracker_{self.n_daps}_unconfirmed_buildings{self.n_daps}gw.csv')
        out['F'] = np.column_stack([-df['rssi'], -df['pdr']])
