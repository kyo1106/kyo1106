#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualization for rough-surface simulations.
Supports:
- Correlation lengths 0–20 step 2.
- Incidence angles 0–70 deg step 10 (phi=0).
Generates:
- 2D scattering heatmaps per angle.
- 3D hemispheres per angle.
- Internal field x-z slices (nosurf/withsurf) per angle.
- Internal field 3D alpha-blend volumes per corr/angle.
All outputs are written to rough_surface_simulation/figures.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import griddata as scipy_griddata


BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CORRELATION_LENGTHS = list(range(0, 21, 2))
INCIDENCE_THETAS = list(range(0, 71, 10))
PROP_UNIT = np.array([0.7071, 0.0, -0.7071])  # used only as default if log missing
NX, NY = 64, 64  # used for internal field reshaping


def load_mueller_data(data_file):
    data = np.loadtxt(data_file, skiprows=1)
    theta = data[:, 0]
    phi = data[:, 1]
    s11 = data[:, 2]
    return theta, phi, s11


def beam_to_lab_direction(theta_beam, phi_beam, prop, inc_pol_x, inc_pol_y):
    theta_rad = np.radians(theta_beam)
    phi_rad = np.radians(phi_beam)
    cos_theta = np.cos(theta_rad)[:, np.newaxis]
    sin_theta = np.sin(theta_rad)[:, np.newaxis]
    cos_phi = np.cos(phi_rad)[:, np.newaxis]
    sin_phi = np.sin(phi_rad)[:, np.newaxis]
    robs_beam = cos_theta * prop + sin_theta * (cos_phi * inc_pol_x + sin_phi * inc_pol_y)
    norm = np.linalg.norm(robs_beam, axis=1, keepdims=True)
    return robs_beam / norm


def direction_to_lab_angles(robs):
    theta_lab = np.degrees(np.arccos(np.clip(robs[:, 2], -1, 1)))
    phi_lab = np.degrees(np.arctan2(robs[:, 1], robs[:, 0]))
    phi_lab[phi_lab < 0] += 360
    return theta_lab, phi_lab


def get_polarization_vectors(log_file):
    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            import re

            pol_x_match = re.search(
                r"Incident polarization X\(per\):\s*\(([-\d.]+),([-\d.]+),([-\d.]+)\)", content
            )
            pol_y_match = re.search(
                r"Incident polarization Y\(par\):\s*\(([-\d.]+),([-\d.]+),([-\d.]+)\)", content
            )
            if pol_x_match and pol_y_match:
                inc_pol_x = np.array(
                    [float(pol_x_match.group(1)), float(pol_x_match.group(2)), float(pol_x_match.group(3))]
                )
                inc_pol_y = np.array(
                    [float(pol_y_match.group(1)), float(pol_y_match.group(2)), float(pol_y_match.group(3))]
                )
                return inc_pol_x, inc_pol_y
    except Exception:
        pass
    return np.array([-0.7071, 0.0, -0.7071]), np.array([0.0, 1.0, 0.0])


def load_internal_field(int_field_x, int_field_y):
    data_x = np.loadtxt(int_field_x, skiprows=1)
    data_y = np.loadtxt(int_field_y, skiprows=1)

    x_coords = data_x[:, 0]
    y_coords = data_x[:, 1]
    z_coords = data_x[:, 2]
    ex = data_x[:, 4] + 1j * data_x[:, 5]
    ey = data_x[:, 6] + 1j * data_x[:, 7]
    ez = data_x[:, 8] + 1j * data_x[:, 9]
    e_intensity = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2

    return x_coords, y_coords, z_coords, e_intensity


def coords_to_indices(coords):
    x_unique = sorted(np.unique(coords[:, 0]))
    y_unique = sorted(np.unique(coords[:, 1]))
    z_unique = sorted(np.unique(coords[:, 2]))
    x_map = {val: idx for idx, val in enumerate(x_unique)}
    y_map = {val: idx for idx, val in enumerate(y_unique)}
    z_map = {val: idx for idx, val in enumerate(z_unique)}

    ix = np.array([x_map[x] for x in coords[:, 0]])
    iy = np.array([y_map[y] for y in coords[:, 1]])
    iz = np.array([z_map[z] for z in coords[:, 2]])

    return ix, iy, iz, len(x_unique), len(y_unique), len(z_unique)


# ------------------------------- scattering -------------------------------- #
print("=" * 70)
print("Scattering visualizations")
print("=" * 70)

for theta_inc in INCIDENCE_THETAS:
    print(f"[Angle {theta_inc} deg] 2D heatmaps...")
    rows = 4
    cols = 3
    fig_2d, axes_2d = plt.subplots(rows, cols, figsize=(20, 24), facecolor="white")
    axes_flat = axes_2d.flatten()

    for idx, corr_len in enumerate(CORRELATION_LENGTHS):
        ax = axes_flat[idx]
        corr_dir = BASE_DIR / f"corr_{corr_len}" / f"angle_{theta_inc}" / "nosurf"
        data_file = corr_dir / "mueller_scatgrid"
        log_file = corr_dir / "log"

        if not data_file.exists():
            ax.text(0.5, 0.5, f"Corr={corr_len}\nNo data", ha="center", va="center")
            ax.set_axis_off()
            continue

        theta_beam, phi_beam, s11 = load_mueller_data(data_file)
        inc_pol_x, inc_pol_y = get_polarization_vectors(log_file)
        robs_lab = beam_to_lab_direction(theta_beam, phi_beam, PROP_UNIT, inc_pol_x, inc_pol_y)
        theta_lab, phi_lab = direction_to_lab_angles(robs_lab)

        mask_upper = theta_lab <= 90
        theta_upper = theta_lab[mask_upper]
        phi_upper = phi_lab[mask_upper]
        s11_upper = s11[mask_upper]

        theta_min, theta_max = theta_upper.min(), theta_upper.max()
        phi_min, phi_max = phi_upper.min(), phi_upper.max()

        n_theta = min(91, len(np.unique(theta_upper)))
        n_phi = min(361, len(np.unique(phi_upper)))

        theta_grid = np.linspace(theta_min, theta_max, n_theta)
        phi_grid = np.linspace(phi_min, phi_max, n_phi)
        theta_grid_mesh, phi_grid_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")

        points = np.column_stack([theta_upper, phi_upper])
        s11_grid = griddata(points, s11_upper, (theta_grid_mesh, phi_grid_mesh), method="linear", fill_value=0)
        s11_norm = (s11_grid - s11_grid.min()) / (s11_grid.max() - s11_grid.min() + 1e-10)

        im = ax.imshow(
            s11_norm,
            aspect="auto",
            origin="lower",
            extent=[0, 360, 0, 90],
            cmap="jet",
            interpolation="bilinear",
        )
        ax.set_title(f"Corr={corr_len}")
        ax.set_xlabel("Phi")
        ax.set_ylabel("Theta")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized S11")

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"01_scattering_2d_angle{theta_inc}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print(f"  saved: 01_scattering_2d_angle{theta_inc}.png")

    print(f"[Angle {theta_inc} deg] 3D hemispheres...")
    fig_3d_hemi = plt.figure(figsize=(20, 24), facecolor="white")
    axes_needed = len(CORRELATION_LENGTHS)
    for idx, corr_len in enumerate(CORRELATION_LENGTHS):
        ax = fig_3d_hemi.add_subplot(rows, cols, idx + 1, projection="3d")
        corr_dir = BASE_DIR / f"corr_{corr_len}" / f"angle_{theta_inc}" / "nosurf"
        data_file = corr_dir / "mueller_scatgrid"
        log_file = corr_dir / "log"

        if not data_file.exists():
            ax.text(0.5, 0.5, 0.5, f"Corr={corr_len}\nNo data", ha="center", va="center")
            ax.set_axis_off()
            continue

        theta_beam, phi_beam, s11 = load_mueller_data(data_file)
        inc_pol_x, inc_pol_y = get_polarization_vectors(log_file)
        robs_lab = beam_to_lab_direction(theta_beam, phi_beam, PROP_UNIT, inc_pol_x, inc_pol_y)
        theta_lab, phi_lab = direction_to_lab_angles(robs_lab)

        mask_upper = theta_lab <= 90
        theta_upper = theta_lab[mask_upper]
        phi_upper = phi_lab[mask_upper]
        s11_upper = s11[mask_upper]

        theta_min, theta_max = theta_upper.min(), theta_upper.max()
        phi_min, phi_max = phi_upper.min(), phi_upper.max()
        n_theta = min(91, len(np.unique(theta_upper)))
        n_phi = min(361, len(np.unique(phi_upper)))

        theta_grid = np.linspace(theta_min, theta_max, n_theta)
        phi_grid = np.linspace(phi_min, phi_max, n_phi)
        theta_grid_mesh, phi_grid_mesh = np.meshgrid(theta_grid, phi_grid, indexing="ij")

        points = np.column_stack([theta_upper, phi_upper])
        s11_grid = griddata(points, s11_upper, (theta_grid_mesh, phi_grid_mesh), method="linear", fill_value=0)
        s11_norm = (s11_grid - s11_grid.min()) / (s11_grid.max() - s11_grid.min() + 1e-10)

        theta_surf = np.radians(theta_grid_mesh)
        phi_surf = np.radians(phi_grid_mesh)
        x_surf = np.sin(theta_surf) * np.cos(phi_surf)
        y_surf = np.sin(theta_surf) * np.sin(phi_surf)
        z_surf = np.cos(theta_surf)

        colors = plt.cm.jet(s11_norm)
        ax.plot_surface(
            x_surf,
            y_surf,
            z_surf,
            facecolors=colors,
            rcount=min(200, n_theta),
            ccount=min(200, n_phi),
            shade=False,
            alpha=0.95,
            edgecolor="none",
            antialiased=True,
            linewidth=0,
        )
        ax.set_title(f"Corr={corr_len}", fontsize=10, fontweight="bold")
        ax.view_init(elev=45, azim=-120)
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"02_scattering_3d_hemisphere_angle{theta_inc}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print(f"  saved: 02_scattering_3d_hemisphere_angle{theta_inc}.png")

# ------------------------------ internal field ----------------------------- #
print("\n" + "=" * 70)
print("Internal field visualizations")
print("=" * 70)

for theta_inc in INCIDENCE_THETAS:
    print(f"[Angle {theta_inc} deg] x-z slices...")
    available_corr = [
        c for c in CORRELATION_LENGTHS if (BASE_DIR / f"corr_{c}" / f"angle_{theta_inc}" / "nosurf" / "IntField-X").exists()
    ]
    if not available_corr:
        print(f"  no IntField data for angle {theta_inc}")
        continue

    fig_xz, axes_xz = plt.subplots(len(available_corr), 2, figsize=(12, 2.5 * len(available_corr)), facecolor="white")

    for row_idx, corr_len in enumerate(available_corr):
        corr_dir = BASE_DIR / f"corr_{corr_len}" / f"angle_{theta_inc}"
        nosurf_dir = corr_dir / "nosurf"
        withsurf_dir = corr_dir / "withsurf"

        if not (nosurf_dir / "IntField-X").exists():
            axes_xz[row_idx, 0].text(0.5, 0.5, "No data", ha="center", va="center")
            axes_xz[row_idx, 0].set_axis_off()
            axes_xz[row_idx, 1].set_axis_off()
            continue

        x_coords, y_coords, z_coords, e_intensity_no = load_internal_field(
            nosurf_dir / "IntField-X", nosurf_dir / "IntField-Y"
        )
        coords_no = np.column_stack([x_coords, y_coords, z_coords])
        ix_no, iy_no, iz_no, nx_act, ny_act, nz_act = coords_to_indices(coords_no)

        e_no_3d = np.zeros((nx_act, ny_act, nz_act))
        for i in range(len(ix_no)):
            if 0 <= ix_no[i] < nx_act and 0 <= iy_no[i] < ny_act and 0 <= iz_no[i] < nz_act:
                e_no_3d[int(ix_no[i]), int(iy_no[i]), int(iz_no[i])] = e_intensity_no[i]

        y_slice = ny_act // 2
        slice_no = e_no_3d[:, y_slice, :].T
        slice_no_clean = np.where(
            np.isnan(slice_no) | (slice_no == 0),
            slice_no[slice_no > 0].min() if np.any(slice_no > 0) else 0,
            slice_no,
        )

        axes_xz[row_idx, 0].imshow(
            slice_no_clean,
            aspect="auto",
            origin="lower",
            extent=[0, nx_act, 0, nz_act],
            cmap="jet",
            interpolation="bilinear",
        )
        axes_xz[row_idx, 0].set_title(f"No surf - Corr={corr_len}")
        axes_xz[row_idx, 0].set_ylabel("Z index")
        if row_idx == len(available_corr) - 1:
            axes_xz[row_idx, 0].set_xlabel("X index")

        if (withsurf_dir / "IntField-X").exists():
            x_coords, y_coords, z_coords, e_intensity_with = load_internal_field(
                withsurf_dir / "IntField-X", withsurf_dir / "IntField-Y"
            )
            coords_with = np.column_stack([x_coords, y_coords, z_coords])
            ix_with, iy_with, iz_with, nx_act2, ny_act2, nz_act2 = coords_to_indices(coords_with)

            e_with_3d = np.zeros((nx_act2, ny_act2, nz_act2))
            for i in range(len(ix_with)):
                if 0 <= ix_with[i] < nx_act2 and 0 <= iy_with[i] < ny_act2 and 0 <= iz_with[i] < nz_act2:
                    e_with_3d[int(ix_with[i]), int(iy_with[i]), int(iz_with[i])] = e_intensity_with[i]

            y_slice = ny_act2 // 2
            slice_with = e_with_3d[:, y_slice, :].T
            slice_with_clean = np.where(
                np.isnan(slice_with) | (slice_with == 0),
                slice_with[slice_with > 0].min() if np.any(slice_with > 0) else 0,
                slice_with,
            )

            axes_xz[row_idx, 1].imshow(
                slice_with_clean,
                aspect="auto",
                origin="lower",
                extent=[0, nx_act2, 0, nz_act2],
                cmap="jet",
                interpolation="bilinear",
            )
            axes_xz[row_idx, 1].set_title(f"With surf - Corr={corr_len}")
            if row_idx == len(available_corr) - 1:
                axes_xz[row_idx, 1].set_xlabel("X index")
        else:
            axes_xz[row_idx, 1].text(0.5, 0.5, "No data", ha="center", va="center")
            axes_xz[row_idx, 1].set_axis_off()

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / f"03_internal_field_xz_angle{theta_inc}.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    print(f"  saved: 03_internal_field_xz_angle{theta_inc}.png")

    print(f"[Angle {theta_inc} deg] 3D volumes...")
    for corr_len in available_corr:
        corr_dir = BASE_DIR / f"corr_{corr_len}" / f"angle_{theta_inc}"
        nosurf_dir = corr_dir / "nosurf"

        if not (nosurf_dir / "IntField-X").exists():
            print(f"    skip corr={corr_len}: no IntField files")
            continue

        x_coords, y_coords, z_coords, e_intensity = load_internal_field(
            nosurf_dir / "IntField-X", nosurf_dir / "IntField-Y"
        )
        coords = np.column_stack([x_coords, y_coords, z_coords])
        ix, iy, iz, nx_act, ny_act, nz_act = coords_to_indices(coords)

        e_3d = np.zeros((nx_act, ny_act, nz_act))
        for i in range(len(ix)):
            if 0 <= ix[i] < nx_act and 0 <= iy[i] < ny_act and 0 <= iz[i] < nz_act:
                e_3d[int(ix[i]), int(iy[i]), int(iz[i])] = e_intensity[i]

        fig_3d = plt.figure(figsize=(12, 9), facecolor="white")
        ax = fig_3d.add_subplot(111, projection="3d")

        x_coords_3d, y_coords_3d, z_coords_3d = np.meshgrid(
            np.arange(nx_act), np.arange(ny_act), np.arange(nz_act), indexing="ij"
        )

        layers_to_show = list(range(0, nz_act, max(1, max(nz_act // 15, 1))))

        for z_layer in layers_to_show:
            e_layer = e_3d[:, :, z_layer]
            x_flat = x_coords_3d[:, :, z_layer].flatten()
            y_flat = y_coords_3d[:, :, z_layer].flatten()
            e_flat = e_layer.flatten()

            mask = e_flat > 0
            if not np.any(mask):
                continue

            x_valid = x_flat[mask]
            y_valid = y_flat[mask]
            e_valid = e_flat[mask]

            interp_factor = 2
            x_dense = np.linspace(0, nx_act - 1, nx_act * interp_factor)
            y_dense = np.linspace(0, ny_act - 1, ny_act * interp_factor)
            x_dense_grid, y_dense_grid = np.meshgrid(x_dense, y_dense)

            e_interp = scipy_griddata(
                np.column_stack([x_valid, y_valid]),
                e_valid,
                (x_dense_grid, y_dense_grid),
                method="cubic",
                fill_value=0,
            )

            e_min = e_interp[e_interp > 0].min() if np.any(e_interp > 0) else 0
            e_max = e_interp.max()
            if e_max > e_min:
                e_norm = (e_interp - e_min) / (e_max - e_min)
            else:
                e_norm = np.zeros_like(e_interp)
            colors = plt.cm.jet(e_norm)

            z_surface = np.full_like(x_dense_grid, z_layer)
            ax.plot_surface(
                x_dense_grid,
                y_dense_grid,
                z_surface,
                facecolors=colors,
                alpha=0.25,
                shade=False,
                edgecolor="none",
                linewidth=0,
                antialiased=True,
            )

        ax.set_xlabel("X index")
        ax.set_ylabel("Y index")
        ax.set_zlabel("Z index")
        ax.set_title(f"3D Internal Field - Corr={corr_len}, Angle={theta_inc}")
        ax.set_xlim([0, nx_act - 1])
        ax.set_ylim([0, ny_act - 1])
        ax.set_zlim([0, nz_act - 1])
        ax.view_init(elev=20, azim=-45)

        plt.tight_layout()
        out_name = FIGURES_DIR / f"04_internal_field_3d_corr{corr_len}_angle{theta_inc}.png"
        plt.savefig(out_name, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"    saved: {out_name.name}")

print("\n" + "=" * 70)
print("All visualizations completed.")
print("=" * 70)
print(f"Results saved to: {FIGURES_DIR}/")
