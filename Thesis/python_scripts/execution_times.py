""" This script generates the figures used in the implementation example.
"""

import os
import scipy.stats as ss
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import timeit
import numpy as np
import elfi

import matplotlib
matplotlib.rcParams['text.usetex'] = True


prepath = '/home/givasile/ORwDS/edinburgh-thesis/Thesis/images/chapter4'
np.random.seed(21)


class Prior:
    def rvs(self, size=None, random_state=None):
        # size from (BS,) -> (BS,1)
        if size is not None:
            size = np.concatenate((size, [1]))
        return ss.uniform(loc=-2.5, scale=5).rvs(size=size, random_state=random_state)

    def pdf(self, theta):
        return ss.uniform(loc=-2.5, scale=5).pdf(theta)

    def logpdf(self, theta):
        return ss.uniform(loc=-2.5, scale=5).logpdf(theta)


class Likelihood:
    r"""Implements the distribution
    P(x|theta) = N(x; theta^4, 1)         if theta in [-0.5, 0.5]
                 N(x; theta + 0.5 + 0.5^4 if theta > 0.5
                 N(x; theta - 0.5 + 0.5^4 if theta < 0.5
    """

    def rvs(self, theta, seed=None):
        """Vectorized sampling from likelihood.

        """
        assert isinstance(theta, np.ndarray)
        theta = theta.astype(np.float)
        samples = np.empty_like(theta)

        c = 0.5 - 0.5 ** 4

        tmp_theta = theta[theta <= -0.5]
        samples[theta <= -
                0.5] = ss.norm(loc=-tmp_theta - c, scale=1).rvs(random_state=seed)
        theta[theta <= -0.5] = np.inf

        tmp_theta = theta[theta <= 0.5]
        samples[theta <= 0.5] = ss.norm(
            loc=tmp_theta**4, scale=1).rvs(random_state=seed)
        theta[theta <= 0.5] = np.inf

        tmp_theta = theta[theta < np.inf]
        samples[theta < np.inf] = ss.norm(
            loc=tmp_theta - c, scale=1).rvs(random_state=seed)
        theta[theta < np.inf] = np.inf

        assert np.allclose(theta, np.inf)
        return samples

    def pdf(self, x, theta):
        """Vectorised pdf evaluation.
        """
        assert isinstance(x, np.ndarray)
        assert isinstance(theta, np.ndarray)
        assert theta.ndim == 1
        assert x.ndim == 1

        BS = theta.shape[0]
        N = x.shape[0]
        theta = theta.astype(np.float)

        pdf_eval = np.zeros((BS))
        c = 0.5 - 0.5 ** 4

        def help_func(lim, mode):
            tmp_theta = theta[theta <= lim]
            tmp_theta = np.expand_dims(tmp_theta, -1)
            scale = np.ones_like(tmp_theta)
            if mode == 1:
                pdf_eval[theta <= lim] = np.prod(
                    ss.norm(loc=-tmp_theta - c, scale=scale).pdf(x), 1)
            elif mode == 2:
                pdf_eval[theta <= lim] = np.prod(
                    ss.norm(loc=tmp_theta**4, scale=scale).pdf(x), 1)
            elif mode == 3:
                pdf_eval[theta <= lim] = np.prod(
                    ss.norm(loc=tmp_theta - c, scale=scale).pdf(x), 1)
            theta[theta <= lim] = np.inf

        big_M = 10**7
        help_func(lim=-0.5, mode=1)
        help_func(lim=0.5, mode=2)
        help_func(lim=big_M, mode=3)
        assert np.allclose(theta, np.inf)
        return pdf_eval


def plot(title, x, y, save_path):
    plt.figure()
    plt.title(title)
    plt.plot(x, y, "ro--")
    plt.xlabel(r"$n_1$")
    plt.ylabel(r"$t (sec))$")
    plt.savefig(os.path.join(prepath, save_path),
                bbox_inches="tight")
    plt.show(block=False)


def summary(x):
    assert isinstance(x, np.ndarray)
    if x.ndim == 1:
        return x
    elif x.ndim == 2:
        return np.prod(x, 1)


def create_factor(x):
    """Creates the function g(theta) = L(theta)*prior(theta).

    """
    lik = Likelihood()
    pr = Prior()

    def tmp_func(theta):
        return float(lik.pdf(x, np.array([theta])) * pr.pdf(theta))
    return tmp_func


def approximate_Z(func, a, b):
    """Approximates the partition function with exhaustive integration.
    """
    return integrate.quad(func, a, b)[0]


def create_gt_posterior(likelihood, prior, data, Z):
    """Returns a function that computes the gt posterior
    """
    def tmp_func(theta):
        return likelihood.pdf(data, np.array([theta])) * prior.pdf(np.array([theta])) / Z
    return tmp_func


data = np.array([0.])
dim = data.shape[0]
a = -2.5  # integration left limit
b = 2.5   # integration right limit

likelihood = Likelihood()
prior = Prior()

factor = create_factor(x=data)
Z = approximate_Z(factor, a, b)
gt_posterior_pdf = create_gt_posterior(likelihood, prior, data, Z)


############# DEFINE ELFI MODEL ##################
def simulator(theta, dim, batch_size=10000, random_state=None):
    likelihood = Likelihood()
    theta = np.repeat(theta, dim, -1)
    return likelihood.rvs(theta, seed=random_state)


elfi.new_model("1D_example")
elfi_prior = elfi.Prior(Prior(), name="theta")
elfi_simulator = elfi.Simulator(
    simulator, elfi_prior, dim, observed=np.expand_dims(data, 0), name="simulator")
dist = elfi.Distance('euclidean', elfi_simulator, name="dist")

bounds = [(-2.5, 2.5)]
dim = data.shape[-1]

# Defines the ROMC inference method
romc = elfi.ROMC(dist, bounds)
romc1 = elfi.ROMC(dist, bounds)

############# Gradients ###################
seed = 21
eps = 1
n1 = np.linspace(5, 20, 4)
solve_grad = []
estimate_regions_grad = []
eval_unnorm_grad = []
sample_grad = []
for i, n in enumerate(n1):
    tic = timeit.default_timer()
    romc.solve_problems(n1=int(n), seed=seed,
                        use_bo=False)
    toc = timeit.default_timer()
    solve_grad.append(toc-tic)

    tic = timeit.default_timer()
    romc.estimate_regions(eps=eps, use_surrogate=False)
    toc = timeit.default_timer()
    estimate_regions_grad.append(toc-tic)

    tic = timeit.default_timer()
    romc.eval_unnorm_posterior(np.array([[0]]))
    toc = timeit.default_timer()
    eval_unnorm_grad.append(toc-tic)

    tic = timeit.default_timer()
    romc.sample(n2=20)
    toc = timeit.default_timer()
    sample_grad.append(toc-tic)

plot("Execution time: solve problems (gradients)",
     n1, solve_grad, "exec_solve_gradients.png")

plot("Execution time: construct regions (gradients)",
     n1, estimate_regions_grad, "exec_regions_gradients.png")

plot("Execution time: evaluate unnormalised posterior (gradients)",
     n1, eval_unnorm_grad, "exec_posterior_gradients.png")

plot("Execution time: sampling (gradients)",
     n1, sample_grad, "exec_sample_gradients.png")


############# bayesian ###################
seed = 21
eps = 1
n1 = np.linspace(5, 20, 4)
solve_bo = []
estimate_regions_bo = []
eval_unnorm_bo = []
sample_bo = []
for i, n in enumerate(n1):
    tic = timeit.default_timer()
    romc.solve_problems(n1=int(n), seed=seed,
                        use_bo=True)
    toc = timeit.default_timer()
    solve_bo.append(toc-tic)

    tic = timeit.default_timer()
    romc.estimate_regions(eps=eps, use_surrogate=True)
    toc = timeit.default_timer()
    estimate_regions_bo.append(toc-tic)

    tic = timeit.default_timer()
    romc.eval_unnorm_posterior(np.array([[0]]))
    toc = timeit.default_timer()
    eval_unnorm_bo.append(toc-tic)

    tic = timeit.default_timer()
    romc.sample(n2=20)
    toc = timeit.default_timer()
    sample_bo.append(toc-tic)

plot("Execution time: solve problems (gradients)",
     n1, solve_bo, "exec_solve_bo.png")

plot("Execution time: construct regions (gradients)",
     n1, estimate_regions_bo, "exec_regions_bo.png")

plot("Execution time: evaluate unnormalised posterior (gradients)",
     n1, eval_unnorm_bo, "exec_posterior_bo.png")

plot("Execution time: sampling (gradients)",
     n1, sample_bo, "exec_sample_bo.png")
