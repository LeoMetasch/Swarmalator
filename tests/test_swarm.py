"""Tests for the Swarm class in src/swarm.py."""

import numpy as np
import pytest
from src.swarm import Swarm, FrecMode


class TestSwarmInitialization:
    """Tests for Swarm initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test that Swarm initializes with valid parameters."""
        swarm = Swarm(N=50, dt=0.1, J=1.0, K=0.0, steps=100)
        assert swarm.N == 50
        assert swarm.dt == 0.1
        assert swarm.J == 1.0
        assert swarm.K == 0.0

    def test_array_shapes(self):
        """Test that state arrays have correct shapes after initialization."""
        N = 100
        swarm = Swarm(N=N, dt=0.1, J=1.0, K=0.0, steps=100)
        assert swarm.x_pos.shape == (N,)
        assert swarm.y_pos.shape == (N,)
        assert swarm.phases.shape == (N,)
        assert swarm.vx.shape == (N,)
        assert swarm.vy.shape == (N,)

    def test_invalid_N_raises(self):
        """Test that invalid N values raise AssertionError."""
        with pytest.raises(AssertionError):
            Swarm(N=0, dt=0.1, J=1.0, K=0.0, steps=100)
        with pytest.raises(AssertionError):
            Swarm(N=-5, dt=0.1, J=1.0, K=0.0, steps=100)

    def test_invalid_dt_raises(self):
        """Test that invalid dt values raise AssertionError."""
        with pytest.raises(AssertionError):
            Swarm(N=50, dt=0, J=1.0, K=0.0, steps=100)
        with pytest.raises(AssertionError):
            Swarm(N=50, dt=-0.1, J=1.0, K=0.0, steps=100)

    def test_predator_initialization(self):
        """Test that predator position is initialized when enabled."""
        swarm = Swarm(N=50, dt=0.1, J=1.0, K=0.0, steps=100, predator=True)
        assert hasattr(swarm, 'pred_x')
        assert hasattr(swarm, 'pred_y')
        assert -1 <= swarm.pred_x <= 1
        assert -1 <= swarm.pred_y <= 1


class TestSwarmStep:
    """Tests for the step() method."""

    def test_step_changes_state(self):
        """Test that step() modifies particle positions/phases."""
        np.random.seed(42)
        swarm = Swarm(N=50, dt=0.1, J=1.0, K=-0.5, steps=100)

        x_before = swarm.x_pos.copy()
        y_before = swarm.y_pos.copy()
        theta_before = swarm.phases.copy()

        swarm.step()

        # At least some values should change (unless at equilibrium, which is unlikely)
        assert not np.allclose(swarm.x_pos, x_before) or \
               not np.allclose(swarm.y_pos, y_before) or \
               not np.allclose(swarm.phases, theta_before)

    def test_step_numba_vs_naive_consistency(self):
        """Test that Numba and naive implementations produce similar results."""
        np.random.seed(42)
        swarm_numba = Swarm(N=30, dt=0.1, J=1.0, K=-0.5, steps=100, use_numba=True)

        np.random.seed(42)
        swarm_naive = Swarm(N=30, dt=0.1, J=1.0, K=-0.5, steps=100, use_numba=False)

        # Run a few steps
        for _ in range(5):
            swarm_numba.step()
            swarm_naive.step()

        # Results should be very close (not exact due to floating point)
        assert np.allclose(swarm_numba.x_pos, swarm_naive.x_pos, rtol=1e-5)
        assert np.allclose(swarm_numba.y_pos, swarm_naive.y_pos, rtol=1e-5)
        assert np.allclose(swarm_numba.phases, swarm_naive.phases, rtol=1e-5)


class TestOrderParameters:
    """Tests for order parameter calculations."""

    def test_correlation_order_parameter_bounds(self):
        """Test that S is in [0, 1]."""
        swarm = Swarm(N=50, dt=0.1, J=1.0, K=0.0, steps=100)
        S = swarm._correlation_order_parameter()
        assert 0 <= S <= 1

    def test_synchrony_order_parameter_bounds(self):
        """Test that R is in [0, 1]."""
        swarm = Swarm(N=50, dt=0.1, J=1.0, K=0.0, steps=100)
        R = swarm._synchrony_order_parameter()
        assert 0 <= R <= 1

    def test_velocity_order_parameter_non_negative(self):
        """Test that V and omega are non-negative."""
        swarm = Swarm(N=50, dt=0.1, J=1.0, K=-0.5, steps=100)
        x_prev = swarm.x_pos.copy()
        y_prev = swarm.y_pos.copy()
        theta_prev = swarm.phases.copy()

        swarm.step()

        V, omega = swarm._calculate_velocity_order_parameter(x_prev, y_prev, theta_prev)
        assert V >= 0
        assert omega >= 0

    def test_synchronized_state_has_high_R(self):
        """Test that fully synchronized phases give R ≈ 1."""
        swarm = Swarm(N=50, dt=0.1, J=1.0, K=0.0, steps=100)
        # Force all phases to be equal
        swarm.phases = np.zeros(50)
        R = swarm._synchrony_order_parameter()
        assert R > 0.99


class TestStabilityAnalysis:
    """Tests for the stability_analysis method."""

    def test_stability_analysis_returns_tuple(self):
        """Test that stability_analysis returns expected tuple structure."""
        swarm = Swarm(N=50, dt=0.1, J=1.0, K=0.0, steps=100)
        result = swarm.stability_analysis()

        assert isinstance(result, tuple)
        assert len(result) == 9

        state, S, V, omega, R, best_k, best_sep, best_comp, best_aniso = result
        assert isinstance(state, str)
        assert isinstance(S, float)
        assert isinstance(V, float)
        assert isinstance(omega, float)
        assert isinstance(R, float)


class TestFrequencyModes:
    """Tests for different frequency initialization modes."""

    def test_zero_mode(self):
        """Test ZERO frequency mode initializes all frequencies to 0."""
        swarm = Swarm(N=50, dt=0.1, J=1.0, K=0.0, steps=100, freq_mode=FrecMode.ZERO)
        assert np.allclose(swarm.nat_freq, 0)

    def test_uniform_mode(self):
        """Test UNIFORM frequency mode initializes all frequencies to 1."""
        swarm = Swarm(N=50, dt=0.1, J=1.0, K=0.0, steps=100, freq_mode=FrecMode.UNIFORM)
        assert np.allclose(swarm.nat_freq, 1)

    def test_bimodal_mode(self):
        """Test BIMODAL frequency mode splits frequencies into +1 and -1."""
        swarm = Swarm(N=50, dt=0.1, J=1.0, K=0.0, steps=100, freq_mode=FrecMode.BIMODAL)
        assert np.all((swarm.nat_freq == 1) | (swarm.nat_freq == -1))
