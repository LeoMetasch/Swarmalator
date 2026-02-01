"""Swarmalator simulation source package.

This package contains the core Swarmalator simulation code, including:
- swarm: The main Swarm class with simulation logic
- run: Simulation runner with parameter sweeps
- plots: Visualization functions for order parameters
- transient_times: Transient time analysis utilities
- benchmark: Performance benchmarking tools
- step: Minimal step function demo
"""
from .swarm import Swarm

__all__ = ["Swarm"]
