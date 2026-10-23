"""Lagrangian flowtracks from JHTDB — interactive 3D visualization.

A marimo notebook that seeds tracer particles in a JHTDB dataset, advects them
with :class:`gyverny.lagrangian.LagrangianTracker` (each step is one
``GetVelocity`` call to JHTDB), and visualizes the resulting flowtracks in an
interactive Plotly 3D viewer.

Trajectories are persisted to **Zarr** so they can be re-loaded (and shared
with flowtracks/openPTV tooling) without re-querying JHTDB.

Run with:

    marimo edit notebooks/lagrangian_flowtracks.py
"""

import marimo as mo

__generated_with__ = "0.25.2"

app = mo.App()

###############################################################################
# Cell 1 — imports
###############################################################################
@app.cell
def __():
    import marimo as mo

    import numpy as np
    import plotly.graph_objects as go
    import plotly.express as px

    from gyverny.jhtdb import JHTDBClient
    from gyverny.lagrangian import LagrangianTracker

    return mo, go, np, px, JHTDBClient, LagrangianTracker


###############################################################################
# Cell 2 — parameters & UI controls
###############################################################################
@app.cell
def __(mo):
    mo.md(
        """
        # Lagrangian flowtracks from JHTDB

        Seed tracer particles in a turbulent flow and advect them through the
        velocity field retrieved from the [Johns Hopkins Turbulence Database
        (JHTDB)](https://turbulence.idies.jhu.edu/). Every integration step
        issues a single ``GetVelocity`` request for all particles at once.

        The default testing token supports up to 4096 points per query. For
        larger runs, request a token from
        `turbulence@lists.johnshopkins.edu`.
        """
    )

    dataset = mo.ui.dropdown(
        options=["isotropic8192", "isotropic4096", "isotropic1024fine", "channel5200"],
        value="isotropic8192",
        label="JHTDB dataset",
    )

    start_time = mo.ui.number(
        value=1, start=1, step=1, label="Start timepoint"
    )
    end_time = mo.ui.number(
        value=4, start=1, step=1, label="End timepoint"
    )
    dt = mo.ui.number(
        value=1.0, start=0.1, step=0.1, label="Time step (dt)"
    )

    spatial_method = mo.ui.dropdown(
        options=["lag4", "lag6", "lag8", "m1q4", "m2q8"],
        value="lag4",
        label="Spatial interpolation",
    )

    n_particles = mo.ui.number(
        value=50, start=1, step=10, label="Number of particles"
    )

    seed_mode = mo.ui.dropdown(
        options=["random", "grid", "manual"],
        value="random",
        label="Seed layout",
    )

    auth_token = mo.ui.text(
        value="edu.jhu.pha.turbulence.testing-201406",
        label="Auth token",
        full_width=True,
    )

    # --- Zarr persistence ---------------------------------------------------
    zarr_path = mo.ui.text(
        value="/tmp/jhtdb_trajectories.zarr",
        label="Zarr store path",
        full_width=True,
    )
    save_to_zarr = mo.ui.checkbox(value=False, label="Save to Zarr after run")
    load_from_zarr = mo.ui.checkbox(value=False, label="Load from existing Zarr")
    existing_zarr_path = mo.ui.text(
        value="/tmp/jhtdb_trajectories.zarr",
        label="Existing Zarr path",
        full_width=True,
    )

    mo.md("#### Controls")
    ui = mo.vstack(
        [
            dataset,
            mo.hstack([start_time, end_time, dt], align="start"),
            mo.hstack([spatial_method, n_particles, seed_mode], align="start"),
            auth_token,
            mo.md("#### Zarr persistence"),
            mo.hstack([save_to_zarr, load_from_zarr], align="start"),
            zarr_path,
            existing_zarr_path,
        ]
    )

    return (
        auth_token,
        dt,
        end_time,
        existing_zarr_path,
        load_from_zarr,
        n_particles,
        save_to_zarr,
        seed_mode,
        spatial_method,
        start_time,
        ui,
        zarr_path,
    )


###############################################################################
# Cell 3 — compute (or load) flowtracks via the Lagrangian tracker
###############################################################################
@app.cell
def __(
    mo,
    LagrangianTracker,
    JHTDBClient,
    auth_token,
    dataset,
    dt,
    end_time,
    existing_zarr_path,
    load_from_zarr,
    n_particles,
    save_to_zarr,
    seed_mode,
    spatial_method,
    start_time,
    np,
    zarr_path,
):
    if load_from_zarr.value:
        # --- load from existing Zarr (no JHTDB query) -----------------------
        tracker = LagrangianTracker.load_zarr(
            existing_zarr_path.value,
            client=JHTDBClient(
                dataset_title=dataset.value,
                output_path="/tmp/jhtdb_out",
                auth_token=auth_token.value,
            ),
            spatial_method=spatial_method.value,
        )
        times, trajectory = tracker.get_trajectory()
        mo.md(f"Loaded trajectories from Zarr: `{existing_zarr_path.value}`")
    else:
        # --- seed particle positions ----------------------------------------
        rng = np.random.default_rng(42)
        # UI widgets are passed by reference; read their current values.
        n_particles_val = int(n_particles.value)
        seed_mode_val = seed_mode.value
        if seed_mode_val == "random":
            # isotropic8192 domain approx [0, 2*pi); keep particles well inside.
            start_positions = rng.uniform(0.5, 6.0, size=(n_particles_val, 3))
        elif seed_mode_val == "grid":
            side = int(np.ceil(n_particles_val ** (1 / 3)))
            lin = np.linspace(0.5, 5.0, side)
            grid = np.stack(np.meshgrid(lin, lin, lin, indexing="ij"), -1)
            start_positions = grid.reshape(-1, 3)[:n_particles_val]
        else:
            # fixed manual seed
            start_positions = np.array(
                [[0.5, 0.5, 0.5], [10.0, 10.0, 10.0], [50.0, 50.0, 50.0]],
                dtype=float,
            )[:n_particles_val]
            if len(start_positions) < n_particles_val:
                pad = rng.uniform(
                    0.5, 6.0, size=(n_particles_val - len(start_positions), 3)
                )
                start_positions = np.vstack([start_positions, pad])

        # --- build client + tracker -----------------------------------------
        client = JHTDBClient(
            dataset_title=dataset.value,
            output_path="/tmp/jhtdb_out",
            auth_token=auth_token.value,
        )
        tracker = LagrangianTracker(
            client=client,
            start_positions=start_positions,
            start_time=int(start_time.value),
            dt=float(dt.value),
            spatial_method=spatial_method.value,
        )
        tracker.run(end_time=float(end_time.value), verbose=False)

        times, trajectory = tracker.get_trajectory()  # (T,), (T, N, 3)

        # --- persist to Zarr ------------------------------------------------
        if save_to_zarr.value:
            tracker.save_zarr(zarr_path.value)
            mo.md(f"Saved trajectories to Zarr: `{zarr_path.value}`")

    return start_positions, times, trajectory, tracker


###############################################################################
# Cell 4 — viewer widgets (slider + history toggle)
###############################################################################
@app.cell
def __(mo, times, trajectory):
    T, N, _ = trajectory.shape

    # Time slider: which snapshot to show.
    time_slider = mo.ui.slider(
        start=0, stop=T - 1, value=T - 1, step=1, label="Snapshot index"
    )
    # Show full trajectory history (faint lines) or just the current snapshot.
    show_history = mo.ui.checkbox(value=True, label="Show full trajectory history")

    return T, N, time_slider, show_history


###############################################################################
# Cell 5 — interactive 3D viewer (Plotly)
###############################################################################
@app.cell
def __(mo, go, np, px, times, trajectory, T, N, time_slider, show_history):
    t_idx = int(time_slider.value)

    fig = go.Figure()

    colors = px.colors.qualitative.Bold * (N // len(px.colors.qualitative.Bold) + 1)

    # Full history lines (faint) + current snapshot points.
    if show_history.value:
        for i in range(N):
            fig.add_trace(
                go.Scatter3d(
                    x=trajectory[:, i, 0],
                    y=trajectory[:, i, 1],
                    z=trajectory[:, i, 2],
                    mode="lines",
                    line=dict(color=colors[i % len(colors)], width=2),
                    opacity=0.25,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Current snapshot as markers.
    fig.add_trace(
        go.Scatter3d(
            x=trajectory[t_idx, :, 0],
            y=trajectory[t_idx, :, 1],
            z=trajectory[t_idx, :, 2],
            mode="markers",
            marker=dict(
                size=5,
                color=np.arange(N),
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="particle id"),
            ),
            name=f"t = {times[t_idx]:.2f}",
            hovertemplate=(
                "particle: %{marker.color}<br>"
                + "x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"Lagrangian flowtracks — snapshot at t = {times[t_idx]:.2f}",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700,
    )

    mo.md("### Interactive 3D viewer")
    fig

    return fig


###############################################################################
# Cell 6 — summary table & export
###############################################################################
@app.cell
def __(mo, np, times, trajectory, tracker):
    final = trajectory[-1]
    start = trajectory[0]
    displacement = np.linalg.norm(final - start, axis=1)

    summary = mo.ui.table(
        data={
            "particle": np.arange(len(final)),
            "start_x": np.round(start[:, 0], 3),
            "start_y": np.round(start[:, 1], 3),
            "start_z": np.round(start[:, 2], 3),
            "final_x": np.round(final[:, 0], 3),
            "final_y": np.round(final[:, 1], 3),
            "final_z": np.round(final[:, 2], 3),
            "displacement": np.round(displacement, 3),
        }
    )

    mo.md(
        f"""
        ### Summary

        Snapshots: **{len(times)}** (t = {times[0]:.2f} → {times[-1]:.2f})
        Particles: **{trajectory.shape[1]}**
        Mean displacement: **{float(np.mean(displacement)):.3f}**
        Max displacement: **{float(np.max(displacement)):.3f}**

        Each integration step is a single `GetVelocity` call to JHTDB for all
        particles at once. Trajectories are stored as Zarr arrays
        `times` (T,) and `trajectory` (T, N, 3) for re-loading with
        flowtracks/openPTV tooling.
        """
    )
    summary

    return displacement, final, summary


###############################################################################
# Cell 7 — layout assembly
###############################################################################
@app.cell
def __(mo, ui, fig):
    mo.md("## Controls + Viewer")
    mo.hstack([ui, fig])
    return

