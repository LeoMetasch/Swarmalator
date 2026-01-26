import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

class Swarmalator:
    def __init__(self, N, J, K, dt, x, y, theta, eps=1e-6):
        self.N = N
        self.J = J
        self.K = K
        self.dt = dt
        self.eps = eps

        self.x = x
        self.y = y
        self.theta = theta

    def step(self):
        """
        Update the positions and phases of swarmalators.

        Args:
            x (np.ndarray): x-coordinates of swarmalators
            y (np.ndarray): y-coordinates of swarmalators
            theta (np.ndarray): phases of swarmalators
            dt (float): time step
            J (float): spatial attraction strength
            K (float): phase coupling strength
            eps (float, optional): numerical stability parameter. Defaults to 1e-12.

        Returns:
            x_next, y_next, theta_next (np.ndarray): updated positions and phases
        """
        N = self.N

        # Pairwise position differences: dX[i,j] = x[j]-x[i], same for dY
        dX = self.x[None, :] - self.x[:, None]
        dY = self.y[None, :] - self.y[:, None]

        # Distances
        dist2 = dX*dX + dY*dY
        # exclude i==i by making the diagonal "infinite distance"
        np.fill_diagonal(dist2, np.inf)
        dist = np.sqrt(dist2)

        # Pairwise phase differences: dTheta[i,j] = theta[j]-theta[i]
        dTheta = self.theta[None, :] - self.theta[:, None]
        c = np.cos(dTheta)
        s = np.sin(dTheta)

        # --- xdot, ydot ---
        # common scalar coefficient for each (i,j): (1+J*cos)/dist - 1/dist^2
        coef = (1.0 + self.J*c) / (dist + self.eps) - 1.0 / (dist2 + self.eps)

        xdot = (dX * coef).sum(axis=1) / N
        ydot = (dY * coef).sum(axis=1) / N

        # --- thetadot ---
        thetadot = (self.K * (s / (dist + self.eps)).sum(axis=1)) / N

        # Euler update
        x_next = self.x + self.dt * xdot
        y_next = self.y + self.dt * ydot
        theta_next = self.theta + self.dt * thetadot

        # optional: keep theta in [-pi, pi)
        theta_next = (theta_next + np.pi) % (2*np.pi) - np.pi

        return x_next, y_next, theta_next

    def time_step(self):
        """ Update the positions and phases of swarmalators """
        self.x, self.y, self.theta = self.step()
        return self.x, self.y, self.theta

    def correlation_order_parameter(self):
        """
        Calculate the correlation order parameter S+-.

        Args:
            x (np.ndarray): x-coordinates of swarmalators
            y (np.ndarray): y-coordinates of swarmalators
            theta (np.ndarray): phases of swarmalators

        Returns:
            S_plus (float): correlation order parameter S+ 
            S_minus (float): correlation order parameter S-   
        """
        phi = np.arctan2(self.y, self.x)
        W_plus = np.exp(1j*(phi + self.theta))
        W_minus = np.exp(1j*(phi - self.theta))
        S_plus = np.abs(W_plus.sum()/self.N)
        S_minus = np.abs(W_minus.sum()/self.N)
        return max(S_plus, S_minus)

    def calculate_velocity_order_parameter(self, x_prev, y_prev, theta_prev):
        """
        Calculate the mean spatial velocity (V) and mean phase velocity (Omega) order parameters by 
        calculating the difference between the current and previous state.

        Args:
            x_prev (np.ndarray): x-coordinates of swarmalators at previous time step
            y_prev (np.ndarray): y-coordinates of swarmalators at previous time step
            theta_prev (np.ndarray): phases of swarmalators at previous time step
        Returns:
            V_mean (float): mean spatial velocity
            omega_mean (float): mean phase velocity
        """
        ############ Spatial velocity (V) ####################################
        dx = self.x - x_prev
        dy = self.y - y_prev
        # Euclidean distance
        V = np.sqrt(dx**2 + dy**2)
        V_mean = np.mean(V)

        ############ Phase velocity (Omega) ##################################
        # Use the shortest path to calculate the phase difference - on a circle, 
        # the shortest path between two angles is not always the absolute difference
        dtheta = self.theta - theta_prev
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

        # Calculate the mean phase velocity - how fast the phases (colours) are changing in the simulation
        # without accounting for the direction of change
        omega_i = np.abs(dtheta) / self.dt
        
        # Are the oscillators still oscillating?
        omega_mean = np.mean(omega_i)

        return V_mean, omega_mean

    def synchrony_order_parameter(self):
        """Calculate the Kuramoto order parameter R for the current phases."""
        return float(np.abs(np.mean(np.exp(1j * self.theta))))

    def stability_analysis(self, x_prev=None, y_prev=None, theta_prev=None, advance=True):
        """
        Analyze the stability of the swarmalator system via the order parameters. 
        Eigenvalues require complex analysis, which is beyond the scope of this project.
        Args:
            x_prev, y_prev, theta_prev (np.ndarray | None): previous step state for velocity-based metrics.
            advance (bool): when True, advance one Euler step before computing metrics.
        Returns:
            state (str): stability state of the swarmalator system
            S_parameter (float): correlation order parameter s = 1 - each spatial position corresponds to a specific phase
            V_parameter (float): mean spatial velocity V > 0 - swarmalators are moving
            Omega_parameter (float): mean phase velocity omega > 0 - swarmalators are changing phases 
            R_parameter (): The Synchrony Order Parameter R = 1 - all swarmalators have the same internal phase
        """
        # If no previous snapshot provided, use current state
        if x_prev is None:
            x_prev = self.x.copy()
        if y_prev is None:
            y_prev = self.y.copy()
        if theta_prev is None:
            theta_prev = self.theta.copy()

        # Optionally advance one step before measuring
        if advance:
            self.time_step()

        S_parameter = self.correlation_order_parameter()
        V_parameter, omega_parameter = self.calculate_velocity_order_parameter(x_prev, y_prev, theta_prev)
        R_parameter = self.synchrony_order_parameter()

        # Can't get the splintered phase wave to work
        if S_parameter > 0.9 and V_parameter < 0.01 and omega_parameter < 0.01:
            state = "Static Phase Wave" # High correlation, zero motion 
        
        elif S_parameter > 0.1 and V_parameter >= 0.01 and omega_parameter >= 0.01:
            state = "Active Phase Wave" # Non-zero correlation AND non-zero motion 
            
        elif S_parameter > 0.1 and V_parameter < 0.01 and omega_parameter < 0.01:
            state = "Splintered Phase Wave" # Non-zero correlation but ZERO motion 
            
        elif S_parameter <= 0.1:
            # Distinguish between Async and Sync when correlation S is zero 
            if R_parameter > 0.9:
                state = "Static Sync" # Zero correlation, High global synchrony 
            else:
                state = "Static Async" # Zero correlation, Zero global synchrony 
                
        else:
            state = "Transitioning"

        return state, S_parameter, V_parameter, omega_parameter, R_parameter
        
    def animate(self, steps):
        plt.ion()
        for t in range(steps):
            self.time_step()
            if t % 50 == 0:
                plt.clf()
                plt.scatter(self.x, self.y, c=self.theta, cmap="hsv")
                plt.xlim(-6, 6)
                plt.ylim(-6, 6)
                plt.title(f"J={self.J}, K={self.K}, t = {t}")
                plt.pause(0.01)
        plt.ioff()
        plt.show()

    def run_with_logging(self, steps, log_path, log_interval=1):
        """
        Run the simulation and append order parameters to a CSV log.

        Args:
            steps (int): Number of Euler steps to perform.
            log_path (str | Path): Destination CSV file. Parent dirs are created when needed.
            log_interval (int): Write every `log_interval` steps (1 = every step).
        """
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = ["step", "S", "V", "omega", "R", "J", "K", "N"]
        is_new_file = not log_path.exists()

        # snapshot of previous state for velocity-based metrics
        prev_x, prev_y, prev_theta = self.x.copy(), self.y.copy(), self.theta.copy()

        with log_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if is_new_file:
                writer.writeheader()

            # log initial state (step 0) with zero velocities
            state0, S0, V0, omega0, R0 = self.stability_analysis(prev_x, prev_y, prev_theta, advance=False)
            writer.writerow({
                "step": 0,
                "S": S0,
                "V": V0,
                "omega": omega0,
                "R": R0,
                "J": self.J,
                "K": self.K,
                "N": self.N,
            })

            for step_idx in range(1, steps + 1):
                self.time_step()

                if step_idx % log_interval == 0:
                    state, S, V, omega, R = self.stability_analysis(prev_x, prev_y, prev_theta, advance=False)
                    writer.writerow({
                        "step": step_idx,
                        "S": S,
                        "V": V,
                        "omega": omega,
                        "R": R,
                        "J": self.J,
                        "K": self.K,
                        "N": self.N,
                    })

                prev_x, prev_y, prev_theta = self.x.copy(), self.y.copy(), self.theta.copy()