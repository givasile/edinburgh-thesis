import scipy.stats as ss
import elfi
import numpy as np


def dual_simulator(prior, seed, batch_size=1, random_state=None):
    theta = ss.uniform(-2.5, 5).rvs(size=(batch_size,), random_state=seed)
    samples = np.empty_like(theta)

    c = 0.5 - 0.5 ** 4

    tmp_theta = theta[theta < -0.5]
    samples[theta < -0.5] = ss.norm(loc=tmp_theta + c, scale=1).rvs(random_state=seed)
    theta[theta < -0.5] = np.inf

    tmp_theta = theta[theta < 0.5]
    samples[theta < 0.5] = ss.norm(loc=tmp_theta**4, scale=1).rvs(random_state=seed)
    theta[theta < 0.5] = np.inf

    tmp_theta = theta[theta < np.inf]
    samples[theta < np.inf] = ss.norm(loc=tmp_theta - c, scale=1).rvs(random_state=seed)
    theta[theta < np.inf] = np.inf

    assert np.allclose(theta, np.inf)
    return samples


prior = elfi.Prior("uniform", -2.5, 2.5, name="prior")
sim = elfi.Simulator(dual_simulator, prior, 2)
model = elfi.get_default_model()

# print(sim.generate(batch_size=10000).mean())
print(model.generate(batch_size=10000, seed=10)["sim"].mean())
# print()
# ss.randint(low=1, high=100).rvs(size=(100,))
