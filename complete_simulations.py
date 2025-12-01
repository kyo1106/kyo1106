#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
补全未完成的仿真
"""
import subprocess
import sys
from pathlib import Path
from config import get_adda_executable, get_scat_grid_input, ADDA_GPU_ID

print("="*70)
print("Completing Unfinished Simulations")
print("="*70)

base_dir = Path('rough_surface_simulation')
correlation_lengths = list(range(0, 65, 8))  # 0, 8, 16, ..., 64

lambda_um = 0.532
dpl = 16
m_silicon = 3.5
k_silicon = 0.01
m_air = 1.00001
k_air = 0.0
prop_x, prop_y, prop_z = 0.7071, 0.0, -0.7071  # 45度入射

n_substrate = 3.5
k_substrate = 0.01

def calculate_surface_z(shape_file):
    """根据shape文件计算需要的surface_z值"""
    import numpy as np
    data = np.loadtxt(shape_file, skiprows=2)
    z_coords = data[:, 2]
    z_min = z_coords.min()
    z_max = z_coords.max()
    z_center = (z_min + z_max) / 2
    
    # dipole size
    lambda_um = 0.532
    dpl = 16
    dipole_size = lambda_um / dpl
    
    # 转换为相对于中心的坐标
    z_relative_min = (z_min - z_center) * dipole_size
    
    # surface_z需要大于abs(z_relative_min) + 余量
    surface_z_needed = abs(z_relative_min) + 0.1
    return surface_z_needed

# 检查哪些需要补全
for corr_len in correlation_lengths:
    corr_dir = base_dir / f'corr_{corr_len}'
    nosurf_dir = corr_dir / 'nosurf'
    withsurf_dir = corr_dir / 'withsurf'
    shape_file = corr_dir / f'rough_surface_corr{corr_len}.shape'
    
    # 检查nosurf是否完成
    nosurf_complete = (nosurf_dir / 'mueller_scatgrid').exists() and (nosurf_dir / 'IntField-X').exists()
    
    # 检查withsurf是否完成（需要两个文件都存在且log文件显示成功）
    has_mueller = (withsurf_dir / 'mueller_scatgrid').exists()
    has_int = (withsurf_dir / 'IntField-X').exists()
    log_file = withsurf_dir / 'log'
    has_success = False
    if log_file.exists():
        log_content = log_file.read_text(encoding='utf-8', errors='ignore')
        has_success = 'Total wall time' in log_content or 'Scattered fields' in log_content
    withsurf_complete = has_mueller and has_int and has_success
    
    print(f"\n相关长 = {corr_len}:")
    print(f"  NO surf: {'完成' if nosurf_complete else '未完成'}")
    print(f"  WITH surf: {'完成' if withsurf_complete else '未完成'}")
    
    # 补全nosurf
    if not nosurf_complete and shape_file.exists():
        print(f"  [补全] 运行NO surf仿真...")
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
            '-dir', str(nosurf_dir),
            '-eps', '0.5',
            '-iter', 'qmr2'
        ]
        result_no = subprocess.run(cmd_no, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result_no.returncode != 0:
            print(f"    [ERROR] NO surf仿真失败: {result_no.stderr[:500]}")
        else:
            print(f"    [OK] NO surf仿真完成")
    
    # 补全withsurf
    if not withsurf_complete and shape_file.exists():
        print(f"  [补全] 运行WITH surf仿真...")
        withsurf_dir.mkdir(exist_ok=True)
        
        # 如果log文件存在但为空，或者仿真看起来卡住了，先清理
        log_file = withsurf_dir / 'log'
        if log_file.exists() and log_file.stat().st_size == 0:
            # 检查是否有部分文件，如果有，保留IntField文件（可能正在写入）
            has_partial = (withsurf_dir / 'IntField-X').exists() or (withsurf_dir / 'IntField-Y').exists()
            if not has_partial:
                # 没有部分文件，清理整个目录
                print(f"    清理空的withsurf目录...")
                for f in withsurf_dir.glob('*'):
                    if f.name != 'log':  # 保留log文件
                        f.unlink()
        
        # 动态计算surface_z
        surface_z = calculate_surface_z(shape_file)
        print(f"    计算的surface_z = {surface_z:.3f}")
        
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
            '-dir', str(withsurf_dir),
            '-eps', '0.5',
            '-iter', 'qmr2'
        ]
        print(f"    执行命令: {' '.join(cmd_with[:5])} ...")
        result_with = subprocess.run(cmd_with, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        if result_with.returncode != 0:
            print(f"    [ERROR] WITH surf仿真失败")
            if result_with.stderr:
                print(f"    错误信息: {result_with.stderr[:500]}")
            if result_with.stdout:
                print(f"    输出信息: {result_with.stdout[-500:]}")
        else:
            print(f"    [OK] WITH surf仿真完成")

print("\n" + "="*70)
print("补全完成！")
print("="*70)

