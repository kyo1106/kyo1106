#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
粗糙表面仿真主脚本
Sweep相关长从0到64，间隔8
"""
import subprocess
import sys
import numpy as np
from pathlib import Path
from generate_rough_surface import generate_rough_surface, create_shape_with_rough_surface
from config import get_adda_executable, get_scat_grid_input, ADDA_GPU_ID

print("="*70)
print("Rough Surface Simulation")
print("="*70)

# 参数设置
nx, ny = 64, 64
nz_air = 15  # 空气层数（上层）
nz_silicon = 15  # 硅层数（下层）
rms_height = 5  # RMS高度
correlation_lengths = list(range(0, 65, 8))  # 0, 8, 16, ..., 64

lambda_um = 0.532
dpl = 16
m_silicon = 3.5
k_silicon = 0.01
m_air = 1.00001
k_air = 0.0
prop_x, prop_y, prop_z = 0.7071, 0.0, -0.7071  # 45度入射

base_dir = Path('rough_surface_simulation')
base_dir.mkdir(exist_ok=True)

print(f"\n仿真参数:")
print(f"  尺寸: {nx}×{ny}")
print(f"  空气层: {nz_air}层")
print(f"  硅层: {nz_silicon}层")
print(f"  RMS高度: {rms_height}")
print(f"  相关长: {correlation_lengths}")
print(f"  入射角: 45度")

# 为每个相关长运行仿真
for corr_len in correlation_lengths:
    print(f"\n{'='*70}")
    print(f"相关长 = {corr_len}")
    print(f"{'='*70}")
    
    # 1. 生成粗糙表面
    print(f"\n[1] 生成粗糙表面...")
    height_map = generate_rough_surface(nx, ny, corr_len, rms_height, seed=42)
    print(f"  高度统计: min={height_map.min():.2f}, max={height_map.max():.2f}, RMS={np.std(height_map):.2f}")
    
    # 2. 创建shape文件
    print(f"\n[2] 创建shape文件...")
    corr_dir = base_dir / f'corr_{corr_len}'
    corr_dir.mkdir(exist_ok=True)
    shape_file = corr_dir / f'rough_surface_corr{corr_len}.shape'
    
    total_layers = create_shape_with_rough_surface(
        nx, ny, nz_air, nz_silicon, height_map, shape_file
    )
    
    # 保存高度图
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(height_map, cmap='terrain', origin='lower')
    ax.set_title(f'Rough Surface Height Map\nCorrelation Length={corr_len}, RMS={rms_height}')
    ax.set_xlabel('X (dipole index)')
    ax.set_ylabel('Y (dipole index)')
    plt.colorbar(im, ax=ax, label='Height (dipoles)')
    plt.tight_layout()
    plt.savefig(corr_dir / f'surface_height_corr{corr_len}.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. 运行ADDA仿真（nosurf）
    print(f"\n[3] 运行NO surf仿真...")
    output_dir_nosurf = corr_dir / 'nosurf'
    output_dir_nosurf.mkdir(exist_ok=True)
    
    cmd_no = [
        get_adda_executable(),
        '-gpu', str(ADDA_GPU_ID),
        '-shape', 'read', str(shape_file),
        '-dpl', str(dpl),
        '-m', str(m_silicon), str(k_silicon), str(m_air), str(k_air),
        '-lambda', str(lambda_um),
        '-prop', str(prop_x), str(prop_y), str(prop_z),
        '-store_scat_grid',
        '-scat_grid_inp', get_scat_grid_input(),
        '-store_int_field',
        '-dir', str(output_dir_nosurf),
        '-eps', '0.5',
        '-iter', 'qmr2'
    ]
    
    result_no = subprocess.run(cmd_no, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    if result_no.returncode != 0:
        print(f"  [ERROR] NO surf仿真失败: {result_no.stderr[:500]}")
        continue
    print("  [OK] NO surf仿真完成")
    
    # 4. 运行ADDA仿真（withsurf）
    print(f"\n[4] 运行WITH surf仿真...")
    output_dir_withsurf = corr_dir / 'withsurf'
    output_dir_withsurf.mkdir(exist_ok=True)
    
    # 动态计算surface_z
    data = np.loadtxt(shape_file, skiprows=2)
    z_coords = data[:, 2]
    z_min = z_coords.min()
    z_max = z_coords.max()
    z_center = (z_min + z_max) / 2
    dipole_size = lambda_um / dpl
    z_relative_min = (z_min - z_center) * dipole_size
    surface_z = abs(z_relative_min) + 0.1
    
    print(f"  计算的surface_z = {surface_z:.3f} (z范围=[{z_min:.1f}, {z_max:.1f}])")
    
    n_substrate = 3.5
    k_substrate = 0.01
    
        cmd_with = [
            get_adda_executable(),
            '-gpu', str(ADDA_GPU_ID),
            '-shape', 'read', str(shape_file),
            '-dpl', str(dpl),
            '-m', str(m_silicon), str(k_silicon), str(m_air), str(k_air),
            '-lambda', str(lambda_um),
            '-prop', str(prop_x), str(prop_y), str(prop_z),
            '-surf', str(surface_z), str(n_substrate), str(k_substrate),
            '-store_scat_grid',
            '-scat_grid_inp', get_scat_grid_input(),
            '-store_int_field',
            '-dir', str(output_dir_withsurf),
            '-eps', '0.5',
            '-iter', 'qmr2'
        ]
    
    result_with = subprocess.run(cmd_with, capture_output=True, text=True, encoding='utf-8', errors='ignore')
    if result_with.returncode != 0:
        print(f"  [ERROR] WITH surf仿真失败: {result_with.stderr[:500]}")
        continue
    print("  [OK] WITH surf仿真完成")

print("\n" + "="*70)
print("所有仿真完成！")
print("="*70)

