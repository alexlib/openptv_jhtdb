"""Sample the JHTDB velocity field at a set of points (getData)."""

from __future__ import annotations

import numpy as np

from gyverny.jhtdb import JHTDBClient


def main() -> None:
    client = JHTDBClient(
        dataset_title="isotropic8192",
        output_path="/tmp/jhtdb_out",
        auth_token="edu.jhu.pha.turbulence.testing-201406",
    )

    points = np.array([
        [0.1, 0.2, 0.3],
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
    ], dtype=float)

    vel = client.get_velocity(points, timepoint=1, spatial_method="lag4")
    print("Velocity at points (ux, uy, uz):")
    print(vel)


if __name__ == "__main__":
    main()
