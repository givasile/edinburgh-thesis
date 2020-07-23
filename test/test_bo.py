import elfi
import elfi.methods.bo as bo
from elfi.methods.parameter_inference import BoDetereministic
import matplotlib.pyplot as plt
import numpy as np

from elfi.examples import ma2

parameter_names = ["t1", "t2"]
bounds = {"t1": (-2, 2), "t2": (-2, 2)}
target_name = "d"
acq_noise_var = np.array([.1, .1])
seed = 21

gp_reg = bo.gpy_regression.GPyRegression(parameter_names=parameter_names, bounds=bounds)
lcbsc = bo.acquisition.LCBSC(gp_reg, noise_var=acq_noise_var, seed=seed)


def det_func(x):
    assert x.ndim == 1
    return np.atleast_1d(x[1] + x[0]/100)

pr1 = elfi.Prior("uniform", -2, 4)
pr2 = elfi.Prior("uniform", -2, 4)
sim = elfi.Simulator(lambda x1, x2: x1+x2, pr1, pr2)
pr = elfi.methods.utils.ModelPrior(sim.model)

gp_trainer = BoDetereministic(det_func=det_func,
                              prior=pr,
                              parameter_names=parameter_names,
                              n_evidence = 20,
                              target_name=target_name,
                              bounds = bounds,
                              acq_noise_var = acq_noise_var,
                              target_model=gp_reg,
                              seed=seed)
gp_trainer.fit()

print(gp_trainer.result)
print(gp_trainer.result.x_min)
