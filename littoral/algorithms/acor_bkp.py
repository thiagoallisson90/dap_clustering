import numpy as np
import matplotlib.pyplot as plt

def sphere(x):
    z = np.sum(np.power(x, 2))
    return z

def roulette_wheel_selection(p):
    r = np.random.rand()
    c = np.cumsum(p)
    j = np.argmax(r <= c)
    return j

class Individual:
    def __init__(self):
        self.position = None
        self.cost = None

# Number of Decision Variables
n_var = 10

# Variables Matrix Size
var_size = (1, n_var)

# Decision Variables Lower and Upper Bounds
var_min = -10
var_max = 10

# ACOR Parameters
max_it = 1000
n_pop = 10
n_sample = 40
q = 0.5
zeta = 1

# Create Population Matrix
pop = [Individual() for _ in range(n_pop)]

# Initialize Population Members
for i in range(n_pop):
    # Create Random Solution
    pop[i].position = np.random.uniform(var_min, var_max, size=var_size)
    
    # Evaluation
    pop[i].cost = sphere(pop[i].position)

# Sort Population
pop.sort(key=lambda x: x.cost)

# Update Best Solution Ever Found
best_sol = pop[0]

# Array to Hold Best Cost Values
best_cost = np.zeros(max_it)

# Solution Weights
w = 1 / (np.sqrt(2 * np.pi) * q * n_pop) * np.exp(-0.5 * (((np.arange(n_pop) - 1) / (q * n_pop)) ** 2))

# Selection Probabilities
p = w / np.sum(w)

for it in range(max_it):
    # Means
    s = np.zeros((n_pop, n_var))
    for l in range(n_pop):
        s[l, :] = pop[l].position

    # Standard Deviations
    sigma = np.zeros((n_pop, n_var))
    for l in range(n_pop):
        D = 0
        for r in range(n_pop):
            D = D + np.abs(s[l, :] - s[r, :])
        sigma[l, :] = zeta * D / (n_pop - 1)

    # Create New Population Array
    newpop = [Individual() for _ in range(n_sample)]

    for t in range(n_sample):
        # Initialize Position Matrix
        newpop[t].position = np.zeros(var_size)

        # Solution Construction
        for i in range(n_var):
            # Select Gaussian Kernel
            l = roulette_wheel_selection(p)
            
            # Generate Gaussian Random Variable
            newpop[t].position[0, i] = s[l, i] + sigma[l, i] * np.random.randn()

        # Apply Variable Bounds
        newpop[t].position = np.maximum(newpop[t].position, var_min)
        newpop[t].position = np.minimum(newpop[t].position, var_max)

        # Evaluation
        newpop[t].cost = sphere(newpop[t].position)

    # Merge Main Population (Archive) and New Population (Samples)
    pop = pop + newpop

    # Sort Population
    pop.sort(key=lambda x: x.cost)

    # Delete Extra Members
    pop = pop[:n_pop]

    # Update Best Solution Ever Found
    best_sol = pop[0]

    # Store Best Cost
    best_cost[it] = best_sol.cost

    # Show Iteration Information
    print(f'Iteration {it+1}: Best Cost = {best_cost[it]}')

# Results
plt.figure()
plt.semilogy(best_cost, linewidth=2)
plt.title('ACOR')
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.grid(True)
plt.show()
