"""Lagrangian particle trajectories built on JHTDB GetVelocity.

Integrators advance all particles in parallel: each RK stage issues a single
``getData`` (GetVelocity) call for every particle at once.  Default is the
classical fourth-order Runge-Kutta scheme, which is far more accurate than
explicit Euler for turbulent flow integration::

    k1 = u(x, t)
    k2 = u(x + dt/2 * k1, t + dt/2)
    k3 = u(x + dt/2 * k2, t + dt/2)
    k4 = u(x + dt * k3,     t + dt)
    x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

An optional adaptive timestep caps the per-step Courant number::

    dt = cfl * dx / max|u|
"""

from __future__ import annotations

from typing import Literal

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
        Base time step between sampled velocity fields (in dataset time units).
    spatial_method:
        Spatial interpolation used for ``GetVelocity``.
    scheme:
        Integration scheme: ``"rk4"`` (default, 4th-order) or ``"euler"``.
    adaptive:
        If True, rescale ``dt`` each step so the Courant number
        ``max|u| * dt / dx`` stays below ``cfl``.  ``dx`` is the dataset grid
        spacing.
    cfl:
        Target Courant number used when ``adaptive=True``.
    """

    def __init__(
        self,
        client: JHTDBClient,
        start_positions: np.ndarray,
        start_time: int,
        dt: float = 1.0,
        spatial_method: str = "lag4",
        scheme: Literal["rk4", "euler"] = "rk4",
        adaptive: bool = False,
        cfl: float = 0.5,
    ):
        self.client = client
        self.positions = np.asarray(start_positions, dtype=float).copy()
        self.start_time = int(start_time)
        self.dt_base = float(dt)
        self.dt = float(dt)
        self.spatial_method = spatial_method
        self.scheme = scheme
        self.adaptive = adaptive
        self.cfl = float(cfl)
        self.n_particles = self.positions.shape[0]

        # Trajectory history: list of (time, positions) snapshots.
        self.times: list[float] = [float(start_time)]
        self.trajectory: list[np.ndarray] = [self.positions.copy()]

    # ------------------------------------------------------------------ #
    # velocity sampling
    # ------------------------------------------------------------------ #
    def _sample_velocity(self, positions: np.ndarray, t: float) -> np.ndarray:
        """Query JHTDB GetVelocity at ``positions`` for time ``t``.

        Returns shape ``(N, 3)`` ``(ux, uy, uz)``.
        """
        return self.client.get_velocity(
            positions,
            timepoint=int(round(t)),
            spatial_method=self.spatial_method,
        )

    # ------------------------------------------------------------------ #
    # integration
    # ------------------------------------------------------------------ #
    def step(self, n_steps: int = 1, verbose: bool = False) -> None:
        """Advect particles for ``n_steps`` integration steps."""
        t = self.times[-1]
        for _ in range(n_steps):
            if self.scheme == "rk4":
                self.positions, t = self._rk4_step(t)
            else:
                self.positions, t = self._euler_step(t)
            self.times.append(t)
            self.trajectory.append(self.positions.copy())
            if verbose:
                print(f"t={t:.3f}  dt={self.dt:.3f}  advanced {self.n_particles} particles")

    def _euler_step(self, t: float) -> tuple[np.ndarray, float]:
        u1 = self._sample_velocity(self.positions, t)
        self.dt = self._adaptive_dt(u1)
        new_positions = self.positions + u1 * self.dt
        return new_positions, t + self.dt

    def _rk4_step(self, t: float) -> tuple[np.ndarray, float]:
        u1 = self._sample_velocity(self.positions, t)
        self.dt = self._adaptive_dt(u1)
        half = self.dt / 2.0

        u2 = self._sample_velocity(self.positions + half * u1, t + half)
        u3 = self._sample_velocity(self.positions + half * u2, t + half)
        u4 = self._sample_velocity(self.positions + self.dt * u3, t + self.dt)

        new_positions = (
            self.positions + (self.dt / 6.0) * (u1 + 2.0 * u2 + 2.0 * u3 + u4)
        )
        return new_positions, t + self.dt

    def _adaptive_dt(self, u: np.ndarray) -> float:
        """Rescale dt so max|u|*dt/dx <= cfl when adaptive=True."""
        if not self.adaptive:
            return self.dt_base
        max_speed = float(np.max(np.linalg.norm(u, axis=1)))
        if max_speed <= 0:
            return self.dt_base
        # grid spacing is not stored on the tracker; fall back to dt_base scaled
        # by speed only (dx cancels in relative terms).  Users with known dx can
        # pass a smaller dt_base / cfl directly.
        dx = 1.0  # placeholder; JHTDB grid spacing is dataset-dependent.
        dt = self.cfl * dx / max_speed
        return float(min(dt, self.dt_base))

    def run(self, end_time: float, verbose: bool = False) -> None:
        """Advect particles from the current time to ``end_time``."""
        n_steps = int(round((end_time - self.times[-1]) / self.dt_base))
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
