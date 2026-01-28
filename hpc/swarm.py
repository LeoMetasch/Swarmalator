import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from enum import Enum
import csv
from pathlib import Path
from numpy.typing import NDArray
from typing import Optional, Tuple

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
    predator: bool
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
        phase_coupling: bool = False,
        predator: bool = False,
        hunting_strength: float = 1.0,
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
        self.predator = predator
        self.hunting_strength = hunting_strength
        
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

        if self.predator:
            self.pred_x = np.random.uniform(-1, 1)
            self.pred_y = np.random.uniform(-1, 1)

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

        if self.predator:

            dx_all = self.x_pos - self.pred_x
            dy_all = self.y_pos - self.pred_y
            dist_sq_all = dx_all**2 + dy_all**2
            
            nearest_idx = np.argmin(dist_sq_all)
            
            target_x = self.x_pos[nearest_idx]
            target_y = self.y_pos[nearest_idx]
            
            hunt_dx = target_x - self.pred_x
            hunt_dy = target_y - self.pred_y
            hunt_dist = np.sqrt(hunt_dx**2 + hunt_dy**2) + self.eps
            
            self.pred_x += (hunt_dx / hunt_dist) * self.dt
            self.pred_y += (hunt_dy / hunt_dist) * self.dt

            pred_dx = self.x_pos - self.pred_x
            pred_dy = self.y_pos - self.pred_y

            d_pred2 = pred_dx**2 + pred_dy**2 + self.eps
            d_pred = np.sqrt(d_pred2)

            repulsion_mag = self.hunting_strength / d_pred2
            
            xdot += (pred_dx / d_pred) * repulsion_mag
            ydot += (pred_dy / d_pred) * repulsion_mag

        self.x_pos += xdot * self.dt
        self.y_pos += ydot * self.dt

    def _correlation_order_parameter(self):
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
        phi = np.arctan2(self.y_pos, self.x_pos)
        W_plus = np.exp(1j*(phi + self.phases))
        W_minus = np.exp(1j*(phi - self.phases))
        S_plus = np.abs(W_plus.sum()/self.N)
        S_minus = np.abs(W_minus.sum()/self.N)
        return max(S_plus, S_minus)

    def _calculate_velocity_order_parameter(self, x_prev, y_prev, theta_prev):
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
        dx = self.x_pos - x_prev
        dy = self.y_pos - y_prev
        # Euclidean distance
        V = np.sqrt(dx**2 + dy**2)
        V_mean = np.mean(V)

        ############ Phase velocity (Omega) ##################################
        # Use the shortest path to calculate the phase difference - on a circle, 
        # the shortest path between two angles is not always the absolute difference
        dtheta = self.phases - theta_prev
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

        # Calculate the mean phase velocity - how fast the phases (colours) are changing in the simulation
        # without accounting for the direction of change
        omega_i = np.abs(dtheta) / self.dt
        
        # Are the oscillators still oscillating?
        omega_mean = np.mean(omega_i)

        return V_mean, omega_mean
    
    def _synchrony_order_parameter(self):
        """Calculate the Kuramoto order parameter R for the current phases."""
        return float(np.abs(np.mean(np.exp(1j * self.phases))))
    
    def _circular_kmeans_labels(
        self,
        theta: NDArray[np.floating],
        k: int,
        iters: int = 30,
        seed: int = 0,
    ) -> NDArray[np.int64]:
        """
        K-means clustering on phases (angles) using the embedding theta -> (cos theta, sin theta).

        This avoids problems at the branch cut (e.g. -pi and +pi represent the same angle).

        Args:
            theta: Phase angles (radians), shape (N,).
            k: Number of clusters.
            iters: Maximum number of k-means iterations.
            seed: Random seed for initialization.

        Returns:
            labels: Integer cluster labels in {0..k-1}, shape (N,).
        """
        rng = np.random.default_rng(seed)
        X = np.c_[np.cos(theta), np.sin(theta)]  # (N,2) points on the unit circle

        # Initialize: pick k random points as centers
        cent = X[rng.choice(X.shape[0], size=k, replace=False)].copy()

        for _ in range(iters):
            # Assignment: squared distances to centers (N,k)
            d2 = ((X[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
            lab = np.argmin(d2, axis=1)

            # Update: mean direction per cluster, renormalized to unit length
            new_cent = cent.copy()
            for j in range(k):
                pts = X[lab == j]
                if pts.shape[0] == 0:
                    new_cent[j] = X[rng.integers(0, X.shape[0])]
                else:
                    v = pts.mean(axis=0)
                    n = np.linalg.norm(v)
                    new_cent[j] = v / (n + 1e-12)

            if np.allclose(new_cent, cent):
                break
            cent = new_cent

        return lab.astype(np.int64)

    def _cluster_separation_score(
        self,
        x: NDArray[np.floating],
        y: NDArray[np.floating],
        labels: NDArray[np.integer],
    ) -> float:
        """
        Measure spatial separation of label-defined groups.

        Score = (mean inter-centroid distance) / (mean within-cluster spread).
        Higher values indicate more clearly separated spatial lobes.

        Args:
            x: x-coordinates, shape (N,).
            y: y-coordinates, shape (N,).
            labels: Integer labels defining groups, shape (N,).

        Returns:
            separation: Dimensionless separation score (>= 0).
        """
        k = int(labels.max()) + 1
        if k < 2:
            return 0.0

        pts = np.c_[np.asarray(x, dtype=float), np.asarray(y, dtype=float)]
        cents = np.array([pts[labels == j].mean(axis=0) for j in range(k)])

        # Within-cluster spread (mean distance to centroid)
        spreads = []
        for j in range(k):
            pj = pts[labels == j]
            if pj.shape[0] < 2:
                continue
            spreads.append(np.sqrt(((pj - cents[j]) ** 2).sum(axis=1)).mean())
        within = float(np.mean(spreads)) if spreads else 0.0

        # Between-centroid distances (mean pairwise distance)
        d = []
        for a in range(k):
            for b in range(a + 1, k):
                d.append(np.linalg.norm(cents[a] - cents[b]))
        between = float(np.mean(d)) if d else 0.0

        return between / (within + 1e-12)

    def _phase_compactness(self, labels: NDArray[np.integer]) -> float:
        """
        Compute weighted within-cluster circular coherence of phases.

        For each cluster j:
            r_j = |mean(exp(i*theta_i))| over i in cluster j
        The returned compactness is the size-weighted average of r_j across clusters.

        Args:
            labels: Integer cluster labels, shape (N,).

        Returns:
            compactness: Value in [0, 1]; higher means tighter phase groups within clusters.
        """
        k = int(labels.max()) + 1
        N = int(labels.size)
        if k < 1 or N == 0:
            return 0.0

        z = np.exp(1j * self.phases)
        comp = 0.0
        for j in range(k):
            idx = labels == j
            nj = int(idx.sum())
            if nj == 0:
                continue
            rj = float(np.abs(z[idx].mean()))
            comp += (nj / N) * rj
        return float(comp)

    def _spatial_anisotropy(self, labels: NDArray[np.integer]) -> float:
        """
        Compute weighted mean spatial anisotropy of clusters.

        For each cluster, compute covariance eigenvalues (lmax, lmin) of (x, y) and define:
            anisotropy = sqrt((lmax + tiny) / (lmin + tiny))

        Blobs -> ~1, arcs/filaments -> large.

        Args:
            labels: Integer cluster labels, shape (N,).

        Returns:
            anisotropy: Weighted mean anisotropy across clusters.
        """
        k = int(labels.max()) + 1
        N = int(labels.size)
        if k < 1 or N == 0:
            return 0.0

        pts = np.c_[self.x_pos, self.y_pos]
        anisos = []
        weights = []
        for j in range(k):
            idx = labels == j
            nj = int(idx.sum())
            if nj < 3:
                continue
            pj = pts[idx]
            cov = np.cov(pj.T)
            try:
                w = np.linalg.eigvalsh(cov)
            except np.linalg.LinAlgError:
                continue
            lmin = float(np.min(w))
            lmax = float(np.max(w))
            aniso = float(np.sqrt((lmax + 1e-12) / (lmin + 1e-12)))
            anisos.append(aniso)
            weights.append(nj)

        if not anisos:
            return 0.0

        weights_arr = np.array(weights, dtype=float)
        weights_arr /= weights_arr.sum()
        return float(np.dot(weights_arr, np.array(anisos, dtype=float)))

    def splinter_diagnostics(
        self,
        k_min: int = 2,
        k_max: int = 12,
        min_frac: float = 0.05,
        seed: int = 0,
        k_penalty: float = 1.5,
    ) -> Tuple[int, float, float, float, Optional[NDArray[np.int64]]]:
        """
        Diagnose whether phases form discrete groups that are also spatially separated.

        This combines:
          - spatial separation of phase-defined groups, and
          - within-cluster phase compactness (circular coherence).

        To reduce false positives (e.g. slicing a smooth phase wave into many groups), the
        score penalizes large k and highly anisotropic (arc-like) spatial clusters.

        Args:
            k_min: Minimum number of phase clusters to evaluate.
            k_max: Maximum number of phase clusters to evaluate.
            min_frac: Minimum fraction of points required in each cluster (tiny clusters ignored).
            seed: Random seed for clustering initialization.
            k_penalty: Exponent controlling the penalty for large k in the score.

        Returns:
            best_k: Selected number of phase clusters.
            best_sep: Spatial separation score (higher => clearer lobes).
            best_comp: Phase compactness in [0, 1] (higher => tighter phase groups).
            best_aniso: Spatial anisotropy (higher => more arc/filament-like).
            labels: Cluster labels for best_k, or None if no valid clustering was found.
        """
        best = {"k": 1, "sep": 0.0, "comp": 0.0, "aniso": 0.0, "labels": None}
        theta = self.phases

        for k in range(int(k_min), int(k_max) + 1):
            labels = self._circular_kmeans_labels(theta, k, seed=seed)

            counts = np.bincount(labels, minlength=k)
            if counts.min() < min_frac * theta.size:
                continue

            sep = self._cluster_separation_score(self.x_pos, self.y_pos, labels)
            comp = self._phase_compactness(labels)
            aniso = self._spatial_anisotropy(labels)

            score = (sep * comp) / (k ** k_penalty) / (1.0 + aniso)
            best_score = (best["sep"] * best["comp"]) / (max(best["k"], 1) ** k_penalty) / (1.0 + best["aniso"])
            if score > best_score:
                best = {"k": k, "sep": float(sep), "comp": float(comp), "aniso": float(aniso), "labels": labels}

        return (
            int(best["k"]),
            float(best["sep"]),
            float(best["comp"]),
            float(best["aniso"]),
            best["labels"].astype(np.int64) if best["labels"] is not None else None,
        )

    def stability_analysis(self) -> Tuple[str, float, float, float, int, float, float, float]:
        """
        Analyze the current system state using order parameters and diagnostics.

        Returns:
            state: Human-readable state label.
            S_parameter: Correlation order parameter S.
            V_parameter: Mean spatial velocity magnitude per step.
            omega_parameter: Mean phase angular velocity magnitude (rad/time).
            best_k: Best number of phase clusters (diagnostics).
            best_sep: Spatial separation score of phase clusters.
            best_comp: Within-cluster phase compactness (circular coherence).
            best_aniso: Spatial anisotropy of phase clusters.
        """
        x_prev = self.x_pos.copy()
        y_prev = self.y_pos.copy()
        theta_prev = self.phases.copy()

        self.step()

        S_parameter = self._correlation_order_parameter()
        V_parameter, omega_parameter = self._calculate_velocity_order_parameter(x_prev, y_prev, theta_prev)
        R_parameter = self._synchrony_order_parameter()

        best_k, best_sep, best_comp, best_aniso, _ = None, None, None, None, None
        # --- Static-ish states ---
        if V_parameter < 0.001 and omega_parameter < 0.01:
            if S_parameter > 0.1:
                best_k, best_sep, best_comp, best_aniso, _ = self.splinter_diagnostics(
                    k_min=2, k_max=12, min_frac=0.05, seed=0, k_penalty=1.5
                )
                if (best_sep > 3.0) and (best_comp > 0.85):
                    state = "Splintered Phase Wave"
                else:
                    state = "Static Phase Wave"
            else:
                R = float(np.abs(np.mean(np.exp(1j * self.phases))))
                state = "Static Sync" if R > 0.9 else "Static Async"

        # --- Moving states ---
        elif S_parameter > 0.1 and V_parameter >= 0.001 and omega_parameter >= 0.01:
            state = "Active Phase Wave"

        elif S_parameter <= 0.1:
            state = "Static Sync" if R_parameter > 0.9 else "Static Async"

        else:
            state = "Transitioning"

        return state, float(S_parameter), float(V_parameter), float(omega_parameter), float(R_parameter), best_k, best_sep, best_comp, best_aniso
   
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

        fieldnames = ["step", "S", "V", "omega", "R", "J", "K", "N", "state"]
        is_new_file = not log_path.exists()

        # snapshot of previous state for velocity-based metrics
        prev_x, prev_y, prev_theta = self.x_pos.copy(), self.y_pos.copy(), self.phases.copy()

        with log_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if is_new_file:
                writer.writeheader()

            # log initial state (step 0) with zero velocities
            state0, S0, V0, omega0, R0, best_k0, best_sep0, best_comp0, best_aniso0 = self.stability_analysis()
            writer.writerow({
                "step": 0,
                "S": S0,
                "V": V0,
                "omega": omega0,
                "R": R0,
                "J": self.J,
                "K": self.K,
                "N": self.N,
                "state": state0,
            })

            for step_idx in range(1, steps + 1):
                self.step()

                if step_idx % log_interval == 0:
                    state, S, V, omega, R, best_k, best_sep, best_comp, best_aniso = self.stability_analysis()
                    writer.writerow({
                        "step": step_idx,
                        "S": S,
                        "V": V,
                        "omega": omega,
                        "R": R,
                        "J": self.J,
                        "K": self.K,
                        "N": self.N,
                        "state": state,
                    })

                prev_x, prev_y, prev_theta = self.x_pos.copy(), self.y_pos.copy(), self.phases.copy()
                     
    def animate(self, steps):
        plt.ion()
        for t in range(steps):
            self.step()
            if t % 50 == 0:
                plt.clf()
                plt.scatter(self.x_pos, self.y_pos, c=self.phases, cmap="hsv")
                plt.xlim(-6, 6)
                plt.ylim(-6, 6)
                plt.title(f"J={self.J}, K={self.K}, t = {t}")
                plt.pause(0.01)
        plt.ioff()
        plt.show()

    def simulate_video(
        self,
        filename: str = "swarm_simulation.mp4",
        interval: int = 1,
        dpi: int = 100,
        fps: int = 30,
        predator_start_step: int = None,
        predator_end_step: int = None,
    ) -> None:
        """
        Run simulation and save as an MP4 video.
        
        Args:
            filename: Output filename (e.g., "swarm.mp4")
            interval: Record a frame every `interval` steps.
            dpi: Resolution of the output video.
            fps: Frames per second of the output video.
            predator_start_step: Step at which to spawn predator (None = use current setting).
            predator_end_step: Step at which to remove predator (None = never remove).
        """
        from matplotlib.animation import FFMpegWriter

        metadata = dict(title='Swarmalator Simulation', artist='Matplotlib')
        writer = FFMpegWriter(fps=fps, metadata=metadata)

        # Track whether we're using dynamic predator control
        dynamic_predator = predator_start_step is not None
        if dynamic_predator:
            self.predator = False  # Start without predator

        fig, ax = plt.subplots()
        # Initialize scatter plot for swarmalators
        scat = ax.scatter(self.x_pos, self.y_pos, c=self.phases, cmap="hsv", vmin=-np.pi, vmax=np.pi)
        
        # Initialize predator marker (hidden initially)
        pred_marker, = ax.plot([], [], 'rx', markersize=15, markeredgewidth=3, label='Predator')
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_aspect('equal')
        ax.set_title("Swarmalator Simulation")
        ax.legend(loc='upper right')

        print(f"Starting simulation video recording: {filename}")
        if dynamic_predator:
            print(f"  Predator will spawn at step {predator_start_step}")
            if predator_end_step is not None:
                print(f"  Predator will be removed at step {predator_end_step}")
        
        with writer.saving(fig, filename, dpi=dpi):
            for t in range(self.steps):
                # Dynamic predator control
                if dynamic_predator:
                    if t == predator_start_step:
                        self.predator = True
                        self.pred_x = np.random.uniform(-1, 1)
                        self.pred_y = np.random.uniform(-1, 1)
                        print(f"  [Step {t}] Predator spawned at ({self.pred_x:.2f}, {self.pred_y:.2f})")
                    elif predator_end_step is not None and t == predator_end_step:
                        self.predator = False
                        print(f"  [Step {t}] Predator removed")
                
                self.step()

                if t % interval == 0:
                    # Update swarmalator positions
                    scat.set_offsets(np.column_stack((self.x_pos, self.y_pos)))
                    scat.set_array(self.phases)
                    
                    # Update predator position if enabled
                    if self.predator:
                        pred_marker.set_data([self.pred_x], [self.pred_y])
                    else:
                        pred_marker.set_data([], [])
                    
                    ax.set_title(f"t = {t}")
                    
                    # Grab frame
                    writer.grab_frame()
                    
                    if t % 1000 == 0:
                        print(f"Progress: {t}/{self.steps} steps...")
        
        plt.close(fig)
        print(f"Simulation complete. Video saved as {filename}")
