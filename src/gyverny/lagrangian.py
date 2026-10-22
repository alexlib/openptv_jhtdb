"""Lagrangian particle trajectories built on JHTDB GetVelocity.

The integrator advances all particles in parallel with a single ``getData``
(GetVelocity) call per time step, then updates positions with an explicit
Euler step::

    x_{n+1} = x_n + u(x_n, t_n) * dt
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import zarr

from .jhtdb import JHTDBClient


class LagrangianTracker:
    """Advect tracer particles through a JHTDB velocity field.

    Parameters
    ----------
    client:
        Configured :class:`JHTDBClient`.
    start_positions:
        Initial particle positions, shape ``(N, 3)``.
    start_time:
        Starting integer time index.
    dt:
        Time step between sampled velocity fields (in dataset time units).
    spatial_method:
        Spatial interpolation used for ``GetVelocity``.
    """

    def __init__(self, client: JHTDBClient, start_positions: np.ndarray,
                 start_time: int, dt: float = 1.0,
                 spatial_method: str = "lag4"):
        self.client = client
        self.positions = np.asarray(start_positions, dtype=float).copy()
        self.start_time = int(start_time)
        self.dt = float(dt)
        self.spatial_method = spatial_method
        self.n_particles = self.positions.shape[0]

        # Trajectory history: list of (time, positions) snapshots.
        self.times: list[float] = [float(start_time)]
        self.trajectory: list[np.ndarray] = [self.positions.copy()]

    def step(self, n_steps: int = 1, verbose: bool = False) -> None:
        """Advect particles for ``n_steps`` Euler steps."""
        t = self.times[-1]
        for _ in range(n_steps):
            velocity = self.client.get_velocity(
                self.positions,
                timepoint=int(round(t)),
                spatial_method=self.spatial_method,
            )
            self.positions = self.positions + velocity * self.dt
            t += self.dt
            self.times.append(t)
            self.trajectory.append(self.positions.copy())
            if verbose:
                print(f"t={t:.3f}  advanced {self.n_particles} particles")

    def run(self, end_time: float, verbose: bool = False) -> None:
        """Advect particles from the current time to ``end_time``."""
        n_steps = int(round((end_time - self.times[-1]) / self.dt))
        self.step(n_steps=n_steps, verbose=verbose)

    def get_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(times, positions)``.

        ``times`` has shape ``(T,)`` and ``positions`` has shape
        ``(T, N, 3)`` where ``T`` is the number of snapshots and ``N`` the
        number of particles.
        """
        return np.asarray(self.times), np.asarray(self.trajectory)

    def save_zarr(self, path: str) -> None:
        """Persist ``times`` and ``trajectory`` to a Zarr store.

        The store contains two arrays:

        - ``times``: shape ``(T,)`` float64
        - ``trajectory``: shape ``(T, N, 3)`` float32
        """
        root = zarr.open(path, mode="w")
        root.create_array("times", data=np.asarray(self.times, dtype="f8"))
        root.create_array(
            "trajectory", data=np.asarray(self.trajectory, dtype="f4")
        )

    @classmethod
    def load_zarr(
        cls, path: str, client: JHTDBClient, spatial_method: str = "lag4"
    ) -> "LagrangianTracker":
        """Load a Zarr store into a :class:`LagrangianTracker` (no recompute).

        The returned tracker is seeded from the first snapshot so that further
        ``.step()`` / ``.run()`` calls continue from the stored state.
        """
        root = zarr.open(path, mode="r")
        times = np.asarray(root["times"][:])
        trajectory = np.asarray(root["trajectory"][:])

        tracker = cls(
            client=client,
            start_positions=trajectory[0],
            start_time=int(times[0]),
            spatial_method=spatial_method,
        )
        tracker.times = [float(t) for t in times]
        tracker.trajectory = [trajectory[i] for i in range(len(times))]
        return tracker
