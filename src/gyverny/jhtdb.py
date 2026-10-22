"""Thin wrapper around givernylocal for JHTDB data access."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from givernylocal.turbulence_dataset import turb_dataset
from givernylocal.turbulence_toolkit import getData, getCutout


class JHTDBClient:
    """Convenience client for querying JHTDB via givernylocal.

    Parameters
    ----------
    dataset_title:
        JHTDB dataset code, e.g. ``"isotropic8192"``.
    output_path:
        Directory where givernylocal writes scratch output.
    auth_token:
        JHTDB authorization token. The default testing token supports up to
        4096 points per query.
    """

    DEFAULT_TOKEN = "edu.jhu.pha.turbulence.testing-201406"

    def __init__(self, dataset_title: str, output_path: str = "/tmp/jhtdb_out",
                 auth_token: str = DEFAULT_TOKEN):
        self.dataset_title = dataset_title
        self.output_path = output_path
        self.auth_token = auth_token
        self.cube = turb_dataset(
            dataset_title=dataset_title,
            output_path=output_path,
            auth_token=auth_token,
        )

    def get_velocity(self, points: np.ndarray, timepoint: int,
                     spatial_method: str = "lag4") -> np.ndarray:
        """Sample the velocity field at points for a single timepoint.

        Parameters
        ----------
        points:
            Array of shape ``(N, 3)`` with ``(x, y, z)`` coordinates.
        timepoint:
            Integer time index valid for the dataset.
        spatial_method:
            Spatial interpolation method (default ``"lag4"``).

        Returns
        -------
        np.ndarray
            Array of shape ``(N, 3)`` with ``(ux, uy, uz)``.
        """
        points = np.asarray(points, dtype=float)
        results = getData(
            self.cube,
            var="velocity",
            timepoint_original=timepoint,
            temporal_method="none",
            spatial_method_original=spatial_method,
            spatial_operator="field",
            points=points,
            verbose=False,
        )
        # getData returns a list of DataFrames (one per timepoint); take ux/uy/uz.
        df = results[0]
        return df[["ux", "uy", "uz"]].to_numpy(dtype=float)

    def get_cutout(self, var: str, axes_ranges: Sequence[Sequence[float]],
                   strides: Sequence[int] = (1, 1, 1, 1),
                   timepoint_range: Sequence[int] | None = None):
        """Extract a gridded cutout from JHTDB.

        Parameters
        ----------
        var:
            Variable name, e.g. ``"velocity"``.
        axes_ranges:
            ``[[xmin, xmax], [ymin, ymax], [zmin, zmax], [tmin, tmax]]``.
        strides:
            Sampling stride per axis ``(sx, sy, sz, st)``.
        timepoint_range:
            Optional override ``[tmin, tmax]`` for the time axis.
        """
        axes = np.array(axes_ranges, dtype=float)
        if timepoint_range is not None:
            axes[3] = [float(timepoint_range[0]), float(timepoint_range[1])]
        return getCutout(
            self.cube,
            var=var,
            xyzt_axes_ranges_original=axes,
            xyzt_strides=list(strides),
            verbose=True,
        )
