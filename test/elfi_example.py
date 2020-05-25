import numpy as np
import scipy.stats as ss
import elfi
import graphviz


def simulator(mu, sigma, batch_size=1, N=10, random_state=None):
    """Draws independent samples from univariate Gaussian N(mu,sigma)

    :param mu: mean of univariate Gaussian
    :param sigma: std of univariate Gaussian
    :param batch_size: nof batches
    :param N: nof samples in the batch
    :param random_state: int
    :returns: 
    :rtype: np.array [batch_size x N]

    """
    # converts them to np.array, if they are list
    mu, sigma = np.atleast_1d(mu, sigma)
    
    return ss.norm.rvs(mu[:, None], sigma[:, None], size=(batch_size, N), random_state=random_state)


def mean(y):
    return np.mean(y, axis=1)

def var(y):
    return np.var(y, axis=1)


# Set the generating parameters that we will try to infer
mean0 = 1
std0 = 3

# Generate some data (using a fixed seed here)
y0 = simulator(mean0, std0, batch_size=1, N=100, random_state=2)
print(y0)

# elfi priors
mu = elfi.Prior('uniform', -2, 4, name="mu")
sigma = elfi.Prior('uniform', 1, 4, name="sigma")

# elfi.Simulator = elfi.priors + function
sim = elfi.Simulator(simulator, mu, sigma, observed=y0, name="sim")

# add Summary statistics to the model
S1 = elfi.Summary(mean, sim, name="S1")
S2 = elfi.Summary(var, sim, name="S2")

d = elfi.Distance('euclidean', S1, S2, name="eucl_distance")

# elfi.draw(d)
# print(mean(y0))
# print(var(y0))

rej = elfi.Rejection(d, batch_size=10000, seed=21)
res = rej.sample(1000, threshold=.5)
print(res)

res.plot_marginals()
plt.show()
