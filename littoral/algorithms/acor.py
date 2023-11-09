import numpy as np
import matplotlib.pyplot as plt

""" 
- Code based on ACO for Continuous Domains in MATLAB (Yarpiz)
    * Link: https://yarpiz.com/67/ypea104-acor
"""

###################################
# Continuos Optimization Function #
###################################
def sphere(x):
    return np.sum(np.power(x, 2))


########################################
# Ant class to represent the solutions #
########################################
class Ant:
    def __init__(self):
        self.position = None
        self.cost = None


#########################################
# Class AcoR - ACO for continuos domain #
#########################################
class AcoR:
    def __init__(self, n_var, limits, fn, max_iter=1000,
                 n_pop=10, n_sample=40, q=0.5, zeta=1):
        self.n_var = n_var
        self.var_size = (1, n_var)
        self.var_min = limits[0]
        self.var_max = limits[1]
        self.fn = fn
        self.max_iter = max_iter

        self.n_pop = n_pop
        self.n_sample = n_sample
        self.q = q
        self.zeta = zeta

    def _init_pop(self):
        self.pop = []

        for i in range(self.n_pop):
            ant = Ant()
            self.pop.append(ant)

            ant.position = \
                np.random.uniform(self.var_min, self.var_max, size=self.var_size)
            ant.cost = self.fn(self.pop[i].position)
    
    def _roulette_wheel_selection(self, p):
        r = np.random.rand()
        c = np.cumsum(p)
        return np.argmax(r <= c)

    def run(self):
        self._init_pop()

        self.pop.sort(key=lambda x: x.cost)
        self.best_sol_ = self.pop[0]
        self.best_cost_ = np.zeros(self.max_iter)

        w = 1 / (np.sqrt(2 * np.pi) * self.q * self.n_pop) \
            * np.exp(-0.5 * (((np.arange(self.n_pop) - 1) \
                              / (self.q * self.n_pop)) ** 2))
        p = w / np.sum(w)

        for it in range(self.max_iter):
            s = np.zeros((self.n_pop, self.n_var))
            for l in range(self.n_pop):
                s[l, :] = self.pop[l].position

            sigma = np.zeros((self.n_pop, self.n_var))
            for l in range(self.n_pop):
                D = 0
                for r in range(self.n_pop):
                    D = D + np.abs(s[l, :] - s[r, :])
                sigma[l, :] = self.zeta * D / (self.n_pop - 1)

            new_pop = [Ant() for _ in range(self.n_sample)]

            for t in range(self.n_sample):
                new_pop[t].position = np.zeros(self.var_size)

                for i in range(self.n_var):
                    l = self._roulette_wheel_selection(p)
                    new_pop[t].position[0, i] = s[l, i] + sigma[l, i] * np.random.randn()
                
                new_pop[t].position = np.maximum(new_pop[t].position, self.var_min)
                new_pop[t].position = np.minimum(new_pop[t].position, self.var_max)
                new_pop[t].cost = self.fn(new_pop[t].position)

            self.pop = self.pop + new_pop
            self.pop.sort(key=lambda x: x.cost)
            self.pop = self.pop[:self.n_pop]

            self.best_sol_ = self.pop[0]
            self.best_cost_[it] = self.best_sol_.cost

            print(f'Iteration {it+1}: Best Cost = {self.best_cost_[it]}')

            return self
    
    def plot(self):
        plt.figure()
        plt.semilogy(self.best_cost_, linewidth=2)
        
        plt.title('ACOR')
        plt.xlabel('Iteration')
        plt.ylabel('Best Cost')
        plt.grid(True)

        plt.show()
        plt.clf()
        plt.close('all')

        return self

################################################################################

if __name__ == '__main__':
    aco = AcoR(n_var=10, limits=(-10, 10), fn=sphere, max_iter=1000)
    aco.run().plot()