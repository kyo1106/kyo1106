#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch rough-surface simulations:
- Correlation length: 0-20 step 2.
- Incidence polar angle: 0-70 deg step 10 (phi=0).
- Runs both nosurf and withsurf (substrate) using GPU.
All output and logs are English-only.
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from config import ADDA_GPU_ID, get_adda_executable, get_scat_grid_input
from generate_rough_surface import (
    create_shape_with_rough_surface,
    generate_rough_surface,
    visualize_rough_surface_3d,
)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def print_flush(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


# ------------------------- simulation parameters -------------------------- #
NX = 256
NY = 256
NZ_AIR = 15
NZ_SILICON = 15
RMS_HEIGHT = 5
DEFAULT_CORR = list(range(0, 21, 2))  # 0, 2, ..., 20
DEFAULT_THETAS = list(range(0, 71, 10))  # 0, 10, ..., 70 degrees

LAMBDA_UM = 0.532
DPL = 16
DIPOLE_SIZE = LAMBDA_UM / DPL

M_SILICON = 3.5
K_SILICON = 0.01
M_AIR = 1.00001
K_AIR = 0.0

SUBSTRATE_N = 3.5
SUBSTRATE_K = 0.01


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR

# Optional overrides via environment variables:
#   CORR_LIST="0,2,4"    ANGLE_LIST="0,10"
ENV_CORR = os.environ.get("CORR_LIST")
ENV_THETAS = os.environ.get("ANGLE_LIST")
if ENV_CORR:
    CORRELATION_LENGTHS = [int(x) for x in ENV_CORR.split(",") if x.strip() != ""]
else:
    CORRELATION_LENGTHS = DEFAULT_CORR

if ENV_THETAS:
    INCIDENCE_THETAS = [int(x) for x in ENV_THETAS.split(",") if x.strip() != ""]
else:
    INCIDENCE_THETAS = DEFAULT_THETAS

print_flush("=" * 70)
print_flush("Rough Surface Simulation (GPU, with substrate)")
print_flush("=" * 70)
print_flush("Params:")
print_flush(f"  size: {NX}x{NY}")
print_flush(f"  air layers: {NZ_AIR}, silicon layers: {NZ_SILICON}")
print_flush(f"  RMS height: {RMS_HEIGHT}")
print_flush(f"  correlation lengths: {CORRELATION_LENGTHS}")
print_flush(f"  incidence thetas: {INCIDENCE_THETAS}")
print_flush(f"  lambda: {LAMBDA_UM}, dpl: {DPL}, dipole_size: {DIPOLE_SIZE:.5f}")
print_flush(f"  substrate (n,k): ({SUBSTRATE_N},{SUBSTRATE_K})")


def compute_surface_z(height_map: np.ndarray) -> float:
    """
    Estimate surface_z from mean roughness (matching create_shape_with_rough_surface rounding).
    """
    height_map_int = np.ceil(height_map).astype(int)
    min_height = height_map_int.min()
    height_adj = height_map_int - min_height
    max_rough = int(height_adj.max())
    mean_rough = float(height_adj.mean())
    total_layers = NZ_SILICON + max_rough + NZ_AIR
    z_center = (total_layers - 1) / 2.0
    surface_index = NZ_SILICON + mean_rough
    surface_z = (surface_index - z_center) * DIPOLE_SIZE
    print_flush(
        f"  roughness stats: min={min_height}, max={max_rough}, mean={mean_rough:.3f}, "
        f"total_layers={total_layers}, surface_z={surface_z:.4f}"
    )
    return surface_z, total_layers


def run_adda(command):
    print_flush(f"  cmd: {' '.join(command[:6])} ...")
    result = subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8", errors="ignore"
    )
    return result


for corr_len in CORRELATION_LENGTHS:
    status_corr = Path(f"running_corr_{corr_len}.status")
    status_corr.touch()
    try:
        print_flush("\n" + "-" * 70)
        print_flush(f"Correlation length = {corr_len}")
        print_flush("-" * 70)

        corr_dir = OUTPUT_ROOT / f"corr_{corr_len}"
        corr_dir.mkdir(exist_ok=True)

        # Generate or reuse shape
        shape_file = corr_dir / f"rough_surface_corr{corr_len}.shape"
        height_map = generate_rough_surface(NX, NY, corr_len, RMS_HEIGHT, seed=42)

        surface_z, total_layers_est = compute_surface_z(height_map)

        print_flush("[shape] writing shape file...")
        total_layers_actual = create_shape_with_rough_surface(
            NX, NY, NZ_AIR, NZ_SILICON, height_map, shape_file
        )
        print_flush(f"  shape layers (actual): {total_layers_actual}")

        # Save previews
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(height_map, cmap="terrain", origin="lower")
        ax.set_title(f"Height Map (corr={corr_len}, RMS={RMS_HEIGHT})")
        ax.set_xlabel("X (dipole)")
        ax.set_ylabel("Y (dipole)")
        plt.colorbar(im, ax=ax, label="Height (dipoles)")
        plt.tight_layout()
        plt.savefig(corr_dir / f"surface_height_corr{corr_len}.png", dpi=150, bbox_inches="tight")
        plt.close()

        visualize_rough_surface_3d(height_map, corr_dir / f"model_3d_corr{corr_len}.png", stride=2)
        print_flush("  saved previews.")

        # Loop over incidence angles
        for theta_deg in INCIDENCE_THETAS:
            status_angle = Path(f"running_corr_{corr_len}_angle_{theta_deg}.status")
            status_angle.touch()
            try:
                print_flush(f"\n[angle={theta_deg} deg]")
                theta_rad = math.radians(theta_deg)
                prop_x = math.sin(theta_rad)
                prop_y = 0.0
                prop_z = -math.cos(theta_rad)

                angle_dir = corr_dir / f"angle_{theta_deg}"
                angle_dir.mkdir(exist_ok=True)

                # nosurf
                output_dir_nosurf = angle_dir / "nosurf"
                output_dir_nosurf.mkdir(exist_ok=True)

                cmd_no = [
                    get_adda_executable(),
                    "-gpu",
                    str(ADDA_GPU_ID),
                    "-shape",
                    "read",
                    str(shape_file),
                    "-dpl",
                    str(DPL),
                    "-m",
                    str(M_SILICON),
                    str(K_SILICON),
                    str(M_AIR),
                    str(K_AIR),
                    "-lambda",
                    str(LAMBDA_UM),
                    "-prop",
                    str(prop_x),
                    str(prop_y),
                    str(prop_z),
                    "-store_scat_grid",
                    "-scat_grid_inp",
                    get_scat_grid_input(),
                    "-store_int_field",
                    "-dir",
                    str(output_dir_nosurf),
                    "-eps",
                    "2",
                    "-iter",
                    "qmr2",
                ]
                res_no = run_adda(cmd_no)
                if res_no.returncode != 0:
                    print_flush(f"  [ERROR] NO surf failed: {res_no.stderr[:400]}")
                    continue
                print_flush("  [OK] NO surf done")

                # withsurf
                output_dir_with = angle_dir / "withsurf"
                output_dir_with.mkdir(exist_ok=True)
                cmd_with = [
                    get_adda_executable(),
                    "-gpu",
                    str(ADDA_GPU_ID),
                    "-shape",
                    "read",
                    str(shape_file),
                    "-dpl",
                    str(DPL),
                    "-m",
                    str(M_SILICON),
                    str(K_SILICON),
                    str(M_AIR),
                    str(K_AIR),
                    "-lambda",
                    str(LAMBDA_UM),
                    "-prop",
                    str(prop_x),
                    str(prop_y),
                    str(prop_z),
                    "-surf",
                    str(surface_z),
                    str(SUBSTRATE_N),
                    str(SUBSTRATE_K),
                    "-store_scat_grid",
                    "-scat_grid_inp",
                    get_scat_grid_input(),
                    "-store_int_field",
                    "-dir",
                    str(output_dir_with),
                    "-eps",
                    "2",
                    "-iter",
                    "qmr2",
                ]
                res_with = run_adda(cmd_with)
                if res_with.returncode != 0:
                    print_flush(f"  [ERROR] WITH surf failed: {res_with.stderr[:400]}")
                else:
                    print_flush("  [OK] WITH surf done")
            finally:
                if status_angle.exists():
                    status_angle.unlink()
    finally:
        if status_corr.exists():
            status_corr.unlink()

print_flush("\n" + "=" * 70)
print_flush("All simulations completed!")
print_flush("=" * 70)
