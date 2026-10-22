# gyverny

JHTDB (Johns Hopkins Turbulence Database) client tools built on [`givernylocal`](https://pypi.org/project/givernylocal/).

This folder is the application code. The JHTDB Python library itself is **not** this repo — install it separately:

```bash
python -m pip install --upgrade givernylocal
```

## Contents

- `src/gyverny/jhtdb.py` — thin client wrapper around `givernylocal` (`getData` point sampling, `getCutout` gridded extraction).
- `src/gyverny/lagrangian.py` — Lagrangian trajectory integrator that advects particles using JHTDB `GetVelocity`.
- `examples/getdata_points.py` — sample velocities at points.
- `examples/getcutout_grid.py` — extract a gridded cutout.
- `examples/lagrangian_trajectories.py` — build Lagrangian particle trajectories.

## Auth

For requests under 4096 points the default testing token works:

```
edu.jhu.pha.turbulence.testing-201406
```

For larger or production use, request a token from `turbulence@lists.johnshopkins.edu`.
