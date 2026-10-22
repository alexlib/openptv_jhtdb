"""Extract a gridded cutout from JHTDB (getCutout)."""

from __future__ import annotations

from gyverny.jhtdb import JHTDBClient


def main() -> None:
    client = JHTDBClient(
        dataset_title="isotropic8192",
        output_path="/tmp/jhtdb_out",
        auth_token="edu.jhu.pha.turbulence.testing-201406",
    )

    # [[xmin,xmax],[ymin,ymax],[zmin,zmax],[tmin,tmax]]
    axes_ranges = [[0, 32], [0, 32], [0, 32], [1, 1]]
    strides = (1, 1, 1, 1)

    client.get_cutout("velocity", axes_ranges, strides=strides)
    print("Cutout written to", client.output_path)


if __name__ == "__main__":
    main()
