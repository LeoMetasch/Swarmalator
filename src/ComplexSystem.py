import numpy as np
import matplotlib.pyplot as plt

def time_step(x, y, theta, L, J, K, dt): # messy function
    """time step function, slow bad complexity
    
    Args:
        x (np.ndarray): 1D array of shape (N)
        y (np.ndarray): 1D array of shape (N)
        theta (np.ndarray): 1D array of shape (N)
        L (float): System size
        J (float): Coupling strength between phase similarity and spatial attraction.
        K (float): Coupling strength for phase synchronization.
        dt (float): Euler time step size

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
        - x_update (np.ndarray): Updated x-positions
        - y_update (np.ndarray): Updated y-positions
        - theta_update (np.ndarray): Updated phases
    """    
    N = len(x)

    vx = np.zeros(N)
    vy = np.zeros(N)
    dtheta = np.zeros(N)

    for i in range(N):
        for j in range(N):
            if i != j:
                dx = x[j] - x[i]
                dy = y[j] - y[i]

                dist = np.sqrt(dx*dx + dy*dy) # distance between particles

                dth = theta[j] - theta[i] # phase difference between particles

                rhat_x = dx/dist
                rhat_y = dy/dist

                attraction = 1 + J * np.cos(dth)
                rep_x = dx/(dist**2)
                rep_y = dy/(dist**2)

                vx[i] += attraction * rhat_x - rep_x
                vy[i] += attraction * rhat_y - rep_y

                dtheta[i] += np.sin(theta[j] - theta[i])/dist # phase coupling term

        vx[i] /= N
        vy[i] /= N
        dtheta[i] = K/N * dtheta[i] # scale phase coupling term

    x_update = x + vx * dt
    y_update = y + vy * dt
    theta_update = (theta + dtheta * dt) % (2 * np.pi)

    return x_update, y_update, theta_update

def time_step_fast(x, y, theta, J, K, dt): # vectorized initial function
    """time step function, vectorized

    Args:
        x (np.ndarray): 1D array of shape (N)
        y (np.ndarray): 1D array of shape (N)
        theta (np.ndarray): 1D array of shape (N)
        L (float): System size
        J (float): Coupling strength between phase similarity and spatial attraction.
        K (float): Coupling strength for phase synchronization.
        dt (float): Euler time step size

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]:
        - x_update (np.ndarray): Updated x-positions
        - y_update (np.ndarray): Updated y-positions
        - theta_update (np.ndarray): Updated phases
    """    
    N = len(x)

    dx = x[None, :] - x[:, None] #vectorized difference
    dy = y[None, :] - y[:, None] #vectorized difference

    dist = np.sqrt(dx*dx + dy*dy)

    np.fill_diagonal(dist, np.inf) # to keep them from interacting with themselves

    dth = theta[None, :] - theta[:, None]

    rhat_x = dx/dist
    rhat_y = dy/dist

    attraction = 1 + J * np.cos(dth)

    rep_x = dx/(dist**2)
    rep_y = dy/(dist**2)
    
    vx = (np.mean(attraction * rhat_x - rep_x, axis=1))
    vy = (np.mean(attraction * rhat_y - rep_y, axis=1))

    dtheta = (K/N) * np.sum(np.sin(dth)/dist, axis=1) # phase coupling term

    x_new = x + vx * dt
    y_new = y + vy * dt
    theta_new = (theta + dt * dtheta) % (2 * np.pi)

    return x_new, y_new, theta_new

N = 200
L = 10.0

x = np.random.uniform(-L/2, L/2, N)
y = np.random.uniform(-L/2, L/2, N)
theta = np.random.uniform(-np.pi, np.pi, N)

J = 0.66
K = -.8
dt = 0.1

plt.ion()

steps = 10000

for t in range(steps):
    x, y, theta = time_step_fast(x, y, theta, J, K, dt)

    if t % 50 == 0:
        plt.clf()  
        plt.scatter(x, y, c=theta, cmap="hsv")
        plt.xlim(-3, 3) 
        plt.ylim(-3, 3)
        plt.title(f"t = {t}")
        plt.legend(["Particles colored by phase"])
        plt.pause(0.01)

plt.ioff()
plt.show()