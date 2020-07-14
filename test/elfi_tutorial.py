import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import elfi
import time

def generative_model(t1, t2, nof_points=100, batch_size=1, random_state=None):
    """
    The generative model is a DAG.
    The simulator samples from both observed and unobserved variables,
    with ancestral sampling given the parameters.

    :param t1:
    :param t2:
    :param nof_points:
    :param batch_size:
    :return:
    """
    # for broadcasting reasons
    assert isinstance(t1, np.ndarray)
    assert isinstance(t2, np.ndarray)
    assert t1.shape[0] == batch_size
    assert t2.shape[0] == batch_size

    if len(t1.shape) == 1:
        t1 = np.expand_dims(t1, -1)
    if len(t2.shape) == 1:
        t2 = np.expand_dims(t2, -1)


    # 20 i.i.d. samples Gaussian Samples from w0, w-1, w-2
    samples = np.random.normal(0, 1, size=(batch_size, nof_points + 2))
    w_0 = samples[:, 2:]
    w_minus_1 = samples[:, 1:-1]
    w_minus_2 = samples[:, :-2]

    # y is a deterministic variable given w
    y = w_0 + t1 * w_minus_1 + t2 * w_minus_2
    return y


# create dataset: 1 batch
t1, t2 = np.array([1]), np.array([3])
nof_points = 100
batch_size = 1
data = generative_model(t1, t2, nof_points, batch_size)

# for i in range(data.shape[0]):
#     plt.plot(data[i], "o-")
# plt.show()


################## Model Definition ##########################
model = elfi.get_default_model()

# set prior to all parameters
t1 = elfi.Prior(distribution=ss.uniform(2, 20), name="t1")
t2 = elfi.Prior(distribution=ss.uniform(2, 20), name="t2")

# simulator = how to sample from likelihood
# which Prior r.v. needs for sampling
# output of simulator is the observed data
simulator = elfi.Simulator(generative_model, t1, t2,
                           observed=data,
                           name="simulator")

# summaries
S1 = elfi.Summary(lambda x: np.var(x, 1), simulator, name="summary_mean")

# distance
d = elfi.Distance('euclidean', S1, name="euclidean")

print(simulator.generate().shape)
print(d.generate().shape)


############### Inference ######################
# rej_sampling = elfi.Rejection(d, batch_size=10000, seed=10, output_names=['simulator'])
# res = rej_sampling.sample(n_samples=10000, quantile=.01)
# t = time.process_time()
# print(time.process_time() - t)
#
# plt.figure()
# res.plot_marginals()
# plt.show()


smc = elfi.SMC(d, batch_size=10000, seed=10, output_names=['simulator'])
res = smc.sample(n_samples=10000, thresholds=[25, 10, 5, 3])
t = time.process_time()
print(time.process_time() - t)

plt.figure()
res.plot_marginals()
plt.show()

# rej_sampling = elfi.Rejection(d, batch_size=1000, seed=10)
# t = time.process_time()
# res = rej_sampling.sample(n_samples=1000, quantile=.01)
# print(time.process_time() - t)
#
#
# rej_sampling = elfi.Rejection(d, batch_size=100, seed=10)
# t = time.process_time()
# res = rej_sampling.sample(n_samples=1000, quantile=.01)
# print(time.process_time() - t)
#
#
# rej_sampling = elfi.Rejection(d, batch_size=10, seed=10)
# t = time.process_time()
# res = rej_sampling.sample(n_samples=1000, quantile=.01)
# print(time.process_time() - t)
#
#
# rej_sampling = elfi.Rejection(d, batch_size=1, seed=10)
# t = time.process_time()
# res = rej_sampling.sample(n_samples=1000, quantile=.01)
# print(time.process_time() - t)




