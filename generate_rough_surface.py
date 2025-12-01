#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate rough-surface height maps and shape files.
Uses Gaussian random fields; correlation length controls smoothness.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def generate_rough_surface(nx, ny, correlation_length, rms_height, seed=None):
    """
    Create a rough-surface height map.

    Args:
        nx, ny: surface dimensions (dipoles)
        correlation_length: correlation length in dipoles
        rms_height: RMS height in dipoles
        seed: RNG seed

    Returns:
        height_map: (nx, ny) ndarray of heights
    """
    if seed is not None:
        np.random.seed(seed)

    if correlation_length == 0:
        return np.zeros((nx, ny))

    white_noise = np.random.randn(nx, ny)

    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
    X, Y = np.meshgrid(x, y, indexing="ij")

    sigma = correlation_length / (2 * np.sqrt(2 * np.log(2)))  # FWHM to sigma
    correlation_filter = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    correlation_filter = correlation_filter / np.sum(correlation_filter)

    noise_fft = np.fft.fft2(white_noise)
    filter_fft = np.fft.fft2(np.fft.fftshift(correlation_filter))
    correlated_noise_fft = noise_fft * filter_fft
    correlated_noise = np.real(np.fft.ifft2(correlated_noise_fft))

    current_rms = np.std(correlated_noise)
    if current_rms > 0:
        height_map = correlated_noise * (rms_height / current_rms)
    else:
        height_map = np.zeros_like(correlated_noise)

    return height_map


def create_shape_with_rough_surface(nx, ny, nz_air, nz_silicon, height_map, output_file):
    """
    Build a shape file with rough interface.
    Structure:
      Top: Air (Material 2)
      Interface: Rough surface
      Bottom: Silicon (Material 1)
    """
    height_map_int = np.ceil(height_map).astype(int)
    min_height = height_map_int.min()
    height_map_adjusted = height_map_int - min_height
    max_roughness = height_map_adjusted.max()

    total_layers = nz_silicon + max_roughness + nz_air

    print(f"  Roughness range: 0 to {max_roughness}")
    print(f"  Total layers: {total_layers} (Si base {nz_silicon} + roughness {max_roughness} + air {nz_air})")

    with output_file.open("w", encoding="utf-8") as f:
        f.write(
            f"# Rough surface slab {nx}x{ny}x{total_layers} (Air {nz_air} + Roughness {max_roughness} + Base Si {nz_silicon})\n"
        )
        f.write("Nmat=2\n")  # domain1=Si, domain2=Air

        count = 0
        count_si = 0
        count_air = 0

        for iz in range(total_layers):
            for iy in range(ny):
                for ix in range(nx):
                    silicon_top_z = nz_silicon + height_map_adjusted[ix, iy]
                    if iz < silicon_top_z:
                        f.write(f"{ix} {iy} {iz} 1\n")  # Silicon
                        count_si += 1
                    else:
                        f.write(f"{ix} {iy} {iz} 2\n")  # Air
                        count_air += 1
                    count += 1

        print(f"  Total dipoles: {count}")
        print(f"  Si dipoles: {count_si}")
        print(f"  Air dipoles: {count_air}")

    return total_layers


def visualize_rough_surface_3d(height_map, output_file, stride=4, base_thickness=10):
    """
    Save a voxel-style 3D preview of the rough surface.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    h_small = height_map[::stride, ::stride]
    nx, ny = h_small.shape

    h_int = np.ceil(h_small).astype(int)
    h_min = h_int.min()
    h_adjusted = h_int - h_min + base_thickness

    max_z = h_adjusted.max()
    voxels = np.zeros((nx, ny, max_z), dtype=bool)

    for x in range(nx):
        for y in range(ny):
            z_height = h_adjusted[x, y]
            voxels[x, y, :z_height] = True

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    ax.voxels(voxels, facecolors="#1f77b490", edgecolors="white", linewidth=0.1)
    ax.view_init(elev=30, azim=-45)

    ax.set_title(f"Voxelized Silicon Slab (Step={stride})\nShowing Discrete Layers")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z (layers)")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()


if __name__ == "__main__":
    nx, ny = 256, 256
    correlation_lengths = list(range(0, 33, 4))  # test sweep
    rms_height = 5

    output_dir = Path("rough_surface_simulation")
    output_dir.mkdir(exist_ok=True)

    for corr_len in correlation_lengths:
        print(f"\nGenerating rough surface corr={corr_len}...")
        height_map = generate_rough_surface(nx, ny, corr_len, rms_height, seed=42)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(height_map.T, cmap="terrain", origin="lower")
        ax.set_title(f"Rough Surface Height Map\nCorrelation Length={corr_len}, RMS={rms_height}")
        ax.set_xlabel("X (dipole index)")
        ax.set_ylabel("Y (dipole index)")
        plt.colorbar(im, ax=ax, label="Height (dipoles)")
        plt.tight_layout()
        plt.savefig(output_dir / f"height_map_corr{corr_len}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved height map: height_map_corr{corr_len}.png")

        visualize_rough_surface_3d(height_map, output_dir / f"model_3d_corr{corr_len}.png", stride=2)
        print(f"  saved 3D preview: model_3d_corr{corr_len}.png")

        print(f"  stats: min={height_map.min():.2f}, max={height_map.max():.2f}, RMS={np.std(height_map):.2f}")
