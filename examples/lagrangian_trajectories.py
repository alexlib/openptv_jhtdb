"""Build Lagrangian particle trajectories from JHTDB velocity fields."""

from __future__ import annotations

import numpy as np

from gyverny.jhtdb import JHTDBClient
from gyverny.lagrangian import LagrangianTracker


def main() -> None:
    client = JHTDBClient(
        dataset_title="isotropic8192",
        output_path="/tmp/jhtdb_out",
        auth_token="edu.jhu.pha.turbulence.testing-201406",
    )

    # Seed particles at a few locations. isotropic8192 timepoints are integers [1, 6].
    start_positions = np.array([
        [0.5, 0.5, 0.5],
        [10.0, 10.0, 10.0],
        [50.0, 50.0, 50.0],
    ], dtype=float)

    tracker = LagrangianTracker(
        client=client,
        start_positions=start_positions,
        start_time=1,
        dt=1.0,
        spatial_method="lag4",
    )

    # Advect from t=1 to t=4 (3 Euler steps). Each step is one GetVelocity call.
    tracker.run(end_time=4.0, verbose=True)

    times, traj = tracker.get_trajectory()
    print("\nTrajectory snapshots (times):", times)
    print("Trajectory shape (T, N, 3):", traj.shape)
    print("\nFinal positions:")
    print(traj[-1])

    np.save("/tmp/jhtdb_lagrangian_times.npy", times)
    np.save("/tmp/jhtdb_lagrangian_traj.npy", traj)
    print("\nSaved to /tmp/jhtdb_lagrangian_times.npy and .traj.npy")


if __name__ == "__main__":
    main()
