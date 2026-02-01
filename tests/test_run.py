"""Tests for run.py simulation runner."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.run import run_once


class TestRunOnce:
    """Tests for the run_once function."""

    def test_run_once_returns_csv_string(self):
        """Test that run_once returns a properly formatted CSV string."""
        result = run_once(
            N=20, J=1.0, K=0.0, seed=42,
            dt=0.1, steps=50, burnin=0, sample_every=10
        )
        
        assert isinstance(result, str)
        parts = result.split(',')
        assert len(parts) == 9  # N, J, K, seed, R, S, V, omega, state

    def test_run_once_deterministic_with_seed(self):
        """Test that run_once produces identical results with same seed."""
        result1 = run_once(N=20, J=1.0, K=0.0, seed=42, dt=0.1, steps=50, burnin=0, sample_every=10)
        result2 = run_once(N=20, J=1.0, K=0.0, seed=42, dt=0.1, steps=50, burnin=0, sample_every=10)
        
        assert result1 == result2

    def test_run_once_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        result1 = run_once(N=20, J=1.0, K=0.0, seed=42, dt=0.1, steps=50, burnin=0, sample_every=10)
        result2 = run_once(N=20, J=1.0, K=0.0, seed=123, dt=0.1, steps=50, burnin=0, sample_every=10)
        
        # Results should differ (at least in some columns)
        assert result1 != result2

    def test_run_once_invalid_N_raises(self):
        """Test that invalid N raises AssertionError."""
        with pytest.raises(AssertionError):
            run_once(N=0, J=1.0, K=0.0, seed=42, dt=0.1, steps=50, burnin=0, sample_every=10)

    def test_run_once_invalid_dt_raises(self):
        """Test that invalid dt raises AssertionError."""
        with pytest.raises(AssertionError):
            run_once(N=20, J=1.0, K=0.0, seed=42, dt=0, steps=50, burnin=0, sample_every=10)

    def test_run_once_invalid_sample_every_raises(self):
        """Test that invalid sample_every raises AssertionError."""
        with pytest.raises(AssertionError):
            run_once(N=20, J=1.0, K=0.0, seed=42, dt=0.1, steps=50, burnin=0, sample_every=0)
