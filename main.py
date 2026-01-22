import numpy as np
import matplotlib.pyplot as plt
from src.swarmalator import Swarmalator

N = 100
J = 1 # spatial attraction strength
K = -0.25  # phase coupling strength
dt = 0.1
np.random.seed(1)
eps = 1e-6

x = np.random.uniform(-1, 1, N)
y = np.random.uniform(-1, 1, N)
theta = np.random.uniform(-np.pi, np.pi, N)

swarmalator = Swarmalator(N, J, K, dt, x, y, theta, eps)
swarmalator.animate(10000)

order_parameters = swarmalator.stability_analysis()
print(order_parameters)