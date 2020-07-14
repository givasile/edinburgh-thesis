import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt

# %matplotlib inline
# %precision 2

import logging
logging.basicConfig(level=logging.INFO)

# Set an arbitrary global seed to keep the randomly generated quantities the same
seed = 1
np.random.seed(seed)

import elfi
from elfi.examples import ma2
model = ma2.get_model(seed_obs=seed)

log_d = elfi.Operation(np.log, model['d'])

bolfi = elfi.BOLFI(log_d,
                   batch_size=1,
                   initial_evidence=20,
                   update_interval=10,
                   bounds={'t1':(-2, 2), 't2':(-1, 1)},
                   acq_noise_var=[0.1, 0.1],
                   seed=seed)
bolfi.fit(n_evidence=200)
