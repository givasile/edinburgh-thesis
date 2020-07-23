import GPy
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display

# GPy.plotting.change_plotting_library('plotly')

# data
X = np.random.uniform(-3., 3., (20,1))
Y = np.sin(X) + np.random.randn(20, 1)*0.05

# GP
kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
m = GPy.models.GPRegression(X,Y,kernel)
print(kernel)


kernel.plot()
m.plot()
m.optimize()
m.plot()
plt.show(block=False)

