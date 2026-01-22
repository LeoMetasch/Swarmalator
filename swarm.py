import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from enum import Enum

class FrecMode(Enum):
    ZERO = "zero" # omega_i = 0 for all
    UNIFORM = "uniform" # omega_i = 1 for all
    BIMODAL = "bimodal" # omega_i = +1 (first half), -1 (second half)
    RANDOM = "random" # omega_i ~ U(-1, 1)

class Swarm:
    N: int
    dt: float
    eps: float
    phases: npt.NDArray[np.float64]
    nat_freq: npt.NDArray[np.float64]
    x_pos: npt.NDArray[np.float64]
    y_pos: npt.NDArray[np.float64]
    velocities: npt.NDArray[np.float64]
    chirality: bool
    freq_mode: FrecMode
    J: float
    K: float
    steps: int
    
    def __init__(
        self,
        N: int,
        dt: float,
        J: float,
        K: float,
        steps: int,
        chirality: bool = False,
        freq_mode: FrecMode = FrecMode.ZERO,
        phase_coupling: bool = False
    ) -> None:

        self.N = N
        self.eps = 1e-12
        self.dt = dt
        self.J = J
        self.K = K
        self.steps = steps
        self.freq_mode = freq_mode
        self.chirality = chirality
        self.phase_coupling = phase_coupling
        
        # State Initialization
        self.phases = np.random.uniform(-np.pi, np.pi, N)
        self.nat_freq = self._init_omega(freq_mode)
        self.x_pos = np.random.uniform(-1, 1, N)
        self.y_pos = np.random.uniform(-1, 1, N)
        
        # Velocities (v_i): Vector quantity with x and y components
        if self.chirality:
            self.vx, self.vy = self.update_velocities()
        else:
            self.vx = np.zeros(N)
            self.vy = np.zeros(N)

        # Phase Coupling Parameters (Q)
        if self.phase_coupling:
            self.Q_x, self.Q_theta = self.phase_calc()
        else:
            self.Q_x = 0.0
            self.Q_theta = 0.0

    def _init_omega(self, mode: FrecMode) -> npt.NDArray[np.float64]:
        match mode:
            case FrecMode.ZERO:    return np.zeros(self.N)
            case FrecMode.UNIFORM: return np.ones(self.N)
            case FrecMode.BIMODAL: return np.where(np.arange(self.N) < self.N//2, 1.0, -1.0)
            case FrecMode.RANDOM:  return np.random.uniform(-1, 1, self.N)

    def phase_calc(self) -> tuple[npt.NDArray[np.float64] | float, npt.NDArray[np.float64] | float]:

        if self.freq_mode == FrecMode.ZERO:
            return np.zeros(self.N), np.zeros(self.N)
        
        omega_norm = np.where(self.nat_freq == 0, 1.0, np.abs(self.nat_freq))
        omega_sign = self.nat_freq / omega_norm
        
        s_i = omega_sign[:, None]
        s_j = omega_sign[None, :]
        diff_sign = np.abs(s_j - s_i)
        
        Q_x_mat = (np.pi / 2) * diff_sign
        Q_theta_mat = (np.pi / 4) * diff_sign
        
        return Q_x_mat, Q_theta_mat

    def update_velocities(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        x_vel = self.nat_freq * np.cos(self.phases + np.pi / 2)
        y_vel = self.nat_freq * np.sin(self.phases + np.pi / 2)
        return x_vel, y_vel

    def step(self) -> None:
        dX = self.x_pos[None, :] - self.x_pos[:, None]
        dY = self.y_pos[None, :] - self.y_pos[:, None]

        # Distances
        dist2 = dX*dX + dY*dY
        # exclude i==i by making the diagonal "infinite distance"
        np.fill_diagonal(dist2, np.inf)
        dist = np.sqrt(dist2)

        dTheta = self.phases[None, :] - self.phases[:, None]
        c = np.cos(dTheta - self.Q_x)
        s = np.sin(dTheta - self.Q_theta)

        coef = (1.0 + self.J*c) / (dist + self.eps) - 1.0 / (dist2 + self.eps)

        thetadot = self.nat_freq + (self.K * (s / (dist + self.eps)).sum(axis=1)) / self.N

        self.phases += thetadot * self.dt
        self.phases = (self.phases + np.pi) % (2*np.pi) - np.pi

        self.vx, self.vy = self.update_velocities()

        xdot = self.vx + (dX * coef).sum(axis=1) / self.N
        ydot = self.vy + (dY * coef).sum(axis=1) / self.N

        self.x_pos += xdot * self.dt
        self.y_pos += ydot * self.dt

    def simulate(self) -> None:
        pass

    def simulate_video(
        self,
        filename: str = "swarm_simulation.mp4",
        interval: int = 1,
        dpi: int = 100,
        fps: int = 30
    ) -> None:
        """
        Run simulation and save as an MP4 video.
        
        Args:
            filename: Output filename (e.g., "swarm.mp4")
            interval: Record a frame every `interval` steps.
            dpi: Resolution of the output video.
            fps: Frames per second of the output video.
        """
        from matplotlib.animation import FFMpegWriter

        metadata = dict(title='Swarmalator Simulation', artist='Matplotlib')
        writer = FFMpegWriter(fps=fps, metadata=metadata)

        fig, ax = plt.subplots()
        # Initialize scatter plot
        scat = ax.scatter(self.x_pos, self.y_pos, c=self.phases, cmap="hsv", vmin=-np.pi, vmax=np.pi)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        ax.set_title("Swarmalator Simulation")

        print(f"Starting simulation video recording: {filename}")
        
        with writer.saving(fig, filename, dpi=dpi):
            for t in range(self.steps):
                self.step()

                if t % interval == 0:
                    # Update plot data
                    scat.set_offsets(np.column_stack((self.x_pos, self.y_pos)))
                    scat.set_array(self.phases)
                    ax.set_title(f"t = {t}")
                    
                    # Grab frame
                    writer.grab_frame()
                    
                    if t % 1000 == 0:
                        print(f"Progress: {t}/{self.steps} steps...")
        
        plt.close(fig)
        print(f"Simulation complete. Video saved as {filename}")

swarm = Swarm(N=100, dt=0.1, J=0.9, K=0, steps=1000, chirality=True, freq_mode=FrecMode.BIMODAL, phase_coupling=True)
swarm.simulate_video()

