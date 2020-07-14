import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import scipy.stats as ss


def norm_deterministic(theta, u):
    rv = ss.norm(theta[0], theta[1])
    return rv.rvs(random_state=u)

def norm_rv(theta):
    return ss.norm(theta[0], theta[1]).rvs(random_state=21)

def uniform_deterministic(u):
    rv = ss.uniform(-1, 1)
    return rv.rvs(random_state=u)


# x = []
# for u in range(1, 10000):
#     x.append(uniform_deterministic(u))

# plt.hist(x)
# plt.show()

eps = 0.3
x_total = []
y_total = []
z_total = []
for u in [2]:

    theta0 = np.linspace(-5, 5, 40)
    theta1 = np.linspace(0.1, 10, 40)
    y = []
    for i, th0 in enumerate(theta0):
        y.append([])
        for th1 in theta1:
            y[i].append(norm_deterministic(theta=(th0, th1), u=u))

    X, Y = np.meshgrid(theta0, theta1)
    Z = np.array(y)
    dist = np.abs(Z)

    # plt.figure()
    # ax = plt.axes(projection="3d")
    # plt.title("Normal Distribution as deterministic: f = N(mu, sigma, seed=%d)" % u)
    # ax.plot_surface(X, Y, Z, cmap="viridis")
    # ax.set_xlabel('x = mu')
    # ax.set_ylabel('y = sigma')
    # ax.set_zlabel('z = f(mu, sigma, u=%d)' % u)
    # plt.show(block=False)
    
    x_total.append(X[dist < eps])
    y_total.append(Y[dist < eps])
    z_total.append(Z[dist < eps])


theta0 = np.concatenate(x_total)
theta1 = np.concatenate(y_total)
distances = np.concatenate(z_total)


# predictions
pr = [norm_rv([theta0[i], theta1[i]]) for i in range(len(theta0))]


print(np.mean(pr))
print(np.std(pr))
