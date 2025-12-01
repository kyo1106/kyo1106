#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute recommended surface_z for each correlation length using the same
height-map generation logic as the simulator (mean roughness relative to center).
"""
from pathlib import Path

import numpy as np

from generate_rough_surface import generate_rough_surface


base_dir = Path("rough_surface_simulation")
correlation_lengths = list(range(0, 21, 2))  # 0, 2, ..., 20

# Constants must match run_rough_surface_simulation.py
nx = ny = 256
nz_air = 15
nz_silicon = 15
rms_height = 5
lambda_um = 0.532
dpl = 16
dipole_size = lambda_um / dpl

print("Recommended surface_z per correlation length (mean roughness wrt center):")
print("=" * 70)

for corr_len in correlation_lengths:
    # Re-generate deterministic height map (seed fixed in simulator)
    height_map = generate_rough_surface(nx, ny, corr_len, rms_height, seed=42)
    height_map_int = np.ceil(height_map).astype(int)
    min_height = height_map_int.min()
    height_map_adjusted = height_map_int - min_height
    max_roughness = int(height_map_adjusted.max())
    mean_roughness = float(height_map_adjusted.mean())

    total_layers = nz_silicon + max_roughness + nz_air
    z_center = (total_layers - 1) / 2.0
    surface_index = nz_silicon + mean_roughness
    surface_z = (surface_index - z_center) * dipole_size

    print(
        f"corr_{corr_len:2d}: layers={total_layers:2d}, "
        f"roughness(mean/max)={mean_roughness:.2f}/{max_roughness}, "
        f"surface_z={surface_z:.4f}"
    )

