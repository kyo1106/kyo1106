#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算每个shape文件需要的surface_z值
"""
import numpy as np
from pathlib import Path

base_dir = Path('rough_surface_simulation')
correlation_lengths = list(range(0, 65, 8))

print("计算每个相关长需要的surface_z值：")
print("="*70)

for corr_len in correlation_lengths:
    shape_file = base_dir / f'corr_{corr_len}' / f'rough_surface_corr{corr_len}.shape'
    if not shape_file.exists():
        continue
    
    # 读取shape文件
    data = np.loadtxt(shape_file, skiprows=2)
    z_coords = data[:, 2]
    
    # ADDA会将坐标转换为相对于中心的坐标
    # 如果总层数是nz，z的范围大约是[0, nz-1]
    # 转换为相对于中心：[(0-nz/2)*d, ((nz-1)-nz/2)*d]
    # 其中d是dipole size
    z_min = z_coords.min()
    z_max = z_coords.max()
    nz_total = int(z_max - z_min + 1)
    
    # dipole size (dpl=16, lambda=0.532)
    lambda_um = 0.532
    dpl = 16
    dipole_size = lambda_um / dpl
    
    # 转换为相对于中心的坐标（ADDA会自动做这个转换）
    # z_center = (z_min + z_max) / 2
    # z_relative_min = (z_min - z_center) * dipole_size
    z_center = (z_min + z_max) / 2
    z_relative_min = (z_min - z_center) * dipole_size
    
    # surface_z需要大于abs(z_relative_min) + 一些余量
    surface_z_needed = abs(z_relative_min) + 0.1
    
    print(f"corr_{corr_len:2d}: z范围=[{z_min:.1f}, {z_max:.1f}], 总层数={nz_total}, "
          f"相对中心z_min={z_relative_min:.3f}, 需要surface_z >= {surface_z_needed:.3f}")

