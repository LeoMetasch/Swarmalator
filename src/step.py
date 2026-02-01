"""Minimal Swarmalator step function and demo animation."""

import numpy as np
import matplotlib.pyplot as plt


def step(x: np.ndarray, y: np.ndarray, theta: np.ndarray, dt: float, J: float, K: float, eps: float = 1e-12):
    """Advance swarmalator positions and phases by one Euler step.

    Args:
        x: $x$-coordinates of agents.
        y: $y$-coordinates of agents.
        theta: Phase angles for each agent.
        dt: Time step.
        J: Spatial attraction strength.
        K: Phase coupling strength.
        eps: Small constant to avoid division by zero.

    Returns:
        Tuple of updated arrays: $(x_{t+1}, y_{t+1}, \theta_{t+1})$.
    """
    N = x.size

    # Pairwise position differences: dX[i,j] = x[j]-x[i], same for dY
    dX = x[None, :] - x[:, None]
    dY = y[None, :] - y[:, None]

    # Distances
    dist2 = dX*dX + dY*dY
    # exclude i==i by making the diagonal "infinite distance"
    np.fill_diagonal(dist2, np.inf)
    dist = np.sqrt(dist2)

    # Pairwise phase differences: dTheta[i,j] = theta[j]-theta[i]
    dTheta = theta[None, :] - theta[:, None]
    c = np.cos(dTheta)
    s = np.sin(dTheta)

    # --- xdot, ydot ---
    # common scalar coefficient for each (i,j): (1+J*cos)/dist - 1/dist^2
    coef = (1.0 + J*c) / (dist + eps) - 1.0 / (dist2 + eps)

    xdot = (dX * coef).sum(axis=1) / N
    ydot = (dY * coef).sum(axis=1) / N

    # --- thetadot ---
    thetadot = (K * (s / (dist + eps)).sum(axis=1)) / N

    # Euler update
    x_next = x + dt * xdot
    y_next = y + dt * ydot
    theta_next = theta + dt * thetadot

    # optional: keep theta in [-pi, pi)
    theta_next = (theta_next + np.pi) % (2*np.pi) - np.pi

    return x_next, y_next, theta_next



N = 1000
J = 0.9    # spatial attraction strength
K = 0    # phase coupling strength
dt = 0.1

x = np.random.uniform(-1, 1, N)
y = np.random.uniform(-1, 1, N)
theta = np.random.uniform(-np.pi, np.pi, N)

plt.ion()  # turn on interactive plotting

steps = 5000

for t in range(steps):
    x, y, theta = step(x, y, theta, dt, J, K)

    if t % 10 == 0:
        plt.clf()
        plt.scatter(x, y, c=theta, cmap="hsv")
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        plt.title(f"t = {t}")
        plt.legend(["Particles colored by phase"])
        plt.pause(0.01)

plt.ioff()
plt.show()

