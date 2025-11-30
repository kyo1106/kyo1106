#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
粗糙表面仿真完整可视化
包括散射模式和内部电场的所有可视化
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from scipy.interpolate import griddata as scipy_griddata

print("="*70)
print("Rough Surface Simulation - Complete Visualization")
print("="*70)

base_dir = Path('rough_surface_simulation')
correlation_lengths = list(range(0, 65, 8))  # 0, 8, 16, ..., 64
figures_dir = base_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# 参数
nx, ny = 64, 64

# ========== 1. 散射模式可视化 ==========
print("\n[1] 散射模式可视化...")

def load_mueller_data(data_file):
    """加载Mueller矩阵数据"""
    data = np.loadtxt(data_file, skiprows=1)
    theta = data[:, 0]
    phi = data[:, 1]
    s11 = data[:, 2]
    return theta, phi, s11

def beam_to_lab_direction(theta_beam, phi_beam, prop, incPolX, incPolY):
    """坐标转换：beam frame -> lab frame"""
    theta_rad = np.radians(theta_beam)
    phi_rad = np.radians(phi_beam)
    cos_theta = np.cos(theta_rad)[:, np.newaxis]
    sin_theta = np.sin(theta_rad)[:, np.newaxis]
    cos_phi = np.cos(phi_rad)[:, np.newaxis]
    sin_phi = np.sin(phi_rad)[:, np.newaxis]
    robs_beam = (cos_theta * prop + sin_theta * (cos_phi * incPolX + sin_phi * incPolY))
    norm = np.linalg.norm(robs_beam, axis=1, keepdims=True)
    return robs_beam / norm

def direction_to_lab_angles(robs):
    """方向向量 -> lab frame角度"""
    theta_lab = np.degrees(np.arccos(np.clip(robs[:, 2], -1, 1)))
    phi_lab = np.degrees(np.arctan2(robs[:, 1], robs[:, 0]))
    phi_lab[phi_lab < 0] += 360
    return theta_lab, phi_lab

# 从第一个仿真读取偏振向量
prop_x, prop_y, prop_z = 0.7071, 0.0, -0.7071
prop_unit = np.array([prop_x, prop_y, prop_z]) / np.linalg.norm(np.array([prop_x, prop_y, prop_z]))

# 尝试从log文件读取偏振向量
def get_polarization_vectors(log_file):
    """从log文件读取偏振向量"""
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            import re
            pol_x_match = re.search(r'Incident polarization X\(per\):\s*\(([-\d.]+),([-\d.]+),([-\d.]+)\)', content)
            pol_y_match = re.search(r'Incident polarization Y\(par\):\s*\(([-\d.]+),([-\d.]+),([-\d.]+)\)', content)
            if pol_x_match and pol_y_match:
                incPolX = np.array([float(pol_x_match.group(1)), float(pol_x_match.group(2)), float(pol_x_match.group(3))])
                incPolY = np.array([float(pol_y_match.group(1)), float(pol_y_match.group(2)), float(pol_y_match.group(3))])
                return incPolX, incPolY
    except:
        pass
    # 默认值（45度入射）
    return np.array([-0.7071, 0.0, -0.7071]), np.array([0.0, 1.0, 0.0])

# 1.1 2D热图对比（所有相关长）
print("  生成2D热图对比...")
fig_2d, axes_2d = plt.subplots(3, 3, figsize=(20, 20), facecolor='white')
axes_2d = axes_2d.flatten()

for idx, corr_len in enumerate(correlation_lengths):
    corr_dir = base_dir / f'corr_{corr_len}'
    nosurf_dir = corr_dir / 'nosurf'
    
    if not (nosurf_dir / 'mueller_scatgrid').exists():
        axes_2d[idx].text(0.5, 0.5, f'Corr={corr_len}\nNot Available', 
                         ha='center', va='center', fontsize=12)
        axes_2d[idx].set_title(f'Correlation Length = {corr_len}')
        continue
    
    # 加载数据
    theta_beam, phi_beam, s11 = load_mueller_data(nosurf_dir / 'mueller_scatgrid')
    
    # 读取偏振向量
    log_file = nosurf_dir / 'log'
    incPolX, incPolY = get_polarization_vectors(log_file)
    
    # 转换到lab frame
    robs_lab = beam_to_lab_direction(theta_beam, phi_beam, prop_unit, incPolX, incPolY)
    theta_lab, phi_lab = direction_to_lab_angles(robs_lab)
    
    # 提取上半球
    mask_upper = theta_lab <= 90
    theta_upper = theta_lab[mask_upper]
    phi_upper = phi_lab[mask_upper]
    s11_upper = s11[mask_upper]
    
    # 创建稀疏网格（避免内存问题）
    theta_min, theta_max = theta_upper.min(), theta_upper.max()
    phi_min, phi_max = phi_upper.min(), phi_upper.max()
    
    # 使用合理的网格分辨率
    n_theta = min(91, len(np.unique(theta_upper)))  # 最多91个theta点（0-90度）
    n_phi = min(361, len(np.unique(phi_upper)))     # 最多361个phi点（0-360度）
    
    theta_grid = np.linspace(theta_min, theta_max, n_theta)
    phi_grid = np.linspace(phi_min, phi_max, n_phi)
    THETA_grid, PHI_grid = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    
    # 插值
    points = np.column_stack([theta_upper, phi_upper])
    s11_grid = griddata(points, s11_upper, (THETA_grid, PHI_grid), method='linear', fill_value=0)
    
    # 归一化
    s11_norm = (s11_grid - s11_grid.min()) / (s11_grid.max() - s11_grid.min() + 1e-10)
    
    # 绘制
    im = axes_2d[idx].imshow(s11_norm, aspect='auto', origin='lower',
                            extent=[0, 360, 0, 90], cmap='jet', interpolation='bilinear')
    axes_2d[idx].set_title(f'Correlation Length = {corr_len}')
    axes_2d[idx].set_xlabel('Phi (degrees)')
    axes_2d[idx].set_ylabel('Theta (degrees)')
    plt.colorbar(im, ax=axes_2d[idx], label='Normalized S11')

plt.tight_layout()
plt.savefig(figures_dir / '01_scattering_2d_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    保存: 01_scattering_2d_comparison.png")

# 1.2 3D半球对比（选择几个代表性的相关长）
print("  生成3D半球对比...")
selected_corr = [0, 16, 32, 48, 64]
fig_3d_hemi = plt.figure(figsize=(20, 12), facecolor='white')

for col_idx, corr_len in enumerate(selected_corr):
    corr_dir = base_dir / f'corr_{corr_len}'
    nosurf_dir = corr_dir / 'nosurf'
    
    if not (nosurf_dir / 'mueller_scatgrid').exists():
        continue
    
    # 加载和转换数据
    theta_beam, phi_beam, s11 = load_mueller_data(nosurf_dir / 'mueller_scatgrid')
    log_file = nosurf_dir / 'log'
    incPolX, incPolY = get_polarization_vectors(log_file)
    robs_lab = beam_to_lab_direction(theta_beam, phi_beam, prop_unit, incPolX, incPolY)
    theta_lab, phi_lab = direction_to_lab_angles(robs_lab)
    
    mask_upper = theta_lab <= 90
    theta_upper = theta_lab[mask_upper]
    phi_upper = phi_lab[mask_upper]
    s11_upper = s11[mask_upper]
    
    # 创建稀疏网格
    theta_min, theta_max = theta_upper.min(), theta_upper.max()
    phi_min, phi_max = phi_upper.min(), phi_upper.max()
    n_theta = min(91, len(np.unique(theta_upper)))
    n_phi = min(361, len(np.unique(phi_upper)))
    
    theta_grid = np.linspace(theta_min, theta_max, n_theta)
    phi_grid = np.linspace(phi_min, phi_max, n_phi)
    THETA_grid, PHI_grid = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    
    points = np.column_stack([theta_upper, phi_upper])
    s11_grid = griddata(points, s11_upper, (THETA_grid, PHI_grid), method='linear', fill_value=0)
    s11_norm = (s11_grid - s11_grid.min()) / (s11_grid.max() - s11_grid.min() + 1e-10)
    
    # 3D表面
    THETA_surf = np.radians(THETA_grid)
    PHI_surf = np.radians(PHI_grid)
    X_surf = np.sin(THETA_surf) * np.cos(PHI_surf)
    Y_surf = np.sin(THETA_surf) * np.sin(PHI_surf)
    Z_surf = np.cos(THETA_surf)
    
    colors = plt.cm.jet(s11_norm)
    
    ax = fig_3d_hemi.add_subplot(2, len(selected_corr), col_idx + 1, projection='3d')
    ax.plot_surface(X_surf, Y_surf, Z_surf, facecolors=colors,
                   rcount=min(200, n_theta), ccount=min(200, n_phi),
                   shade=False, alpha=0.95, edgecolor='none', antialiased=True, linewidth=0)
    ax.set_title(f'Corr={corr_len}', fontsize=12, fontweight='bold')
    ax.view_init(elev=90, azim=0)
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([0, 1])
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

plt.tight_layout()
plt.savefig(figures_dir / '02_scattering_3d_hemisphere.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    保存: 02_scattering_3d_hemisphere.png")

# ========== 2. 内部电场可视化 ==========
print("\n[2] 内部电场可视化...")

def load_internal_field(int_field_x, int_field_y):
    """加载内部电场数据"""
    data_x = np.loadtxt(int_field_x, skiprows=1)
    data_y = np.loadtxt(int_field_y, skiprows=1)
    
    x_coords = data_x[:, 0]
    y_coords = data_x[:, 1]
    z_coords = data_x[:, 2]
    Ex = data_x[:, 4] + 1j * data_x[:, 5]
    Ey = data_x[:, 6] + 1j * data_x[:, 7]
    Ez = data_x[:, 8] + 1j * data_x[:, 9]
    E_intensity = np.abs(Ex)**2 + np.abs(Ey)**2 + np.abs(Ez)**2
    
    return x_coords, y_coords, z_coords, E_intensity

def coords_to_indices(coords, nx, ny, nz):
    """坐标转索引"""
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

# 2.1 x-z切面对比
print("  生成x-z切面对比...")
fig_xz, axes_xz = plt.subplots(len(selected_corr), 2, figsize=(16, 4*len(selected_corr)), facecolor='white')

for row_idx, corr_len in enumerate(selected_corr):
    corr_dir = base_dir / f'corr_{corr_len}'
    nosurf_dir = corr_dir / 'nosurf'
    withsurf_dir = corr_dir / 'withsurf'
    
    if not (nosurf_dir / 'IntField-X').exists():
        continue
    
    # NO surf
    x_coords, y_coords, z_coords, E_intensity_no = load_internal_field(
        nosurf_dir / 'IntField-X', nosurf_dir / 'IntField-Y'
    )
    coords_no = np.column_stack([x_coords, y_coords, z_coords])
    ix_no, iy_no, iz_no, nx_act, ny_act, nz_act = coords_to_indices(coords_no, nx, ny, 100)
    
    E_no_3d = np.zeros((nx_act, ny_act, nz_act))
    for i in range(len(ix_no)):
        if 0 <= ix_no[i] < nx_act and 0 <= iy_no[i] < ny_act and 0 <= iz_no[i] < nz_act:
            E_no_3d[int(ix_no[i]), int(iy_no[i]), int(iz_no[i])] = E_intensity_no[i]
    
    y_slice = ny_act // 2
    slice_no = E_no_3d[:, y_slice, :].T
    slice_no_clean = np.where(np.isnan(slice_no) | (slice_no == 0),
                             slice_no[slice_no > 0].min() if np.any(slice_no > 0) else 0,
                             slice_no)
    
    im1 = axes_xz[row_idx, 0].imshow(slice_no_clean, aspect='auto', origin='lower',
                                    extent=[0, nx_act, 0, nz_act], cmap='jet', interpolation='bilinear')
    axes_xz[row_idx, 0].set_title(f'NO surf - Corr={corr_len}')
    axes_xz[row_idx, 0].set_xlabel('X'); axes_xz[row_idx, 0].set_ylabel('Z')
    plt.colorbar(im1, ax=axes_xz[row_idx, 0], label='|E|^2')
    
    # WITH surf
    if (withsurf_dir / 'IntField-X').exists():
        x_coords, y_coords, z_coords, E_intensity_with = load_internal_field(
            withsurf_dir / 'IntField-X', withsurf_dir / 'IntField-Y'
        )
        coords_with = np.column_stack([x_coords, y_coords, z_coords])
        ix_with, iy_with, iz_with, nx_act2, ny_act2, nz_act2 = coords_to_indices(coords_with, nx, ny, 100)
        
        E_with_3d = np.zeros((nx_act2, ny_act2, nz_act2))
        for i in range(len(ix_with)):
            if 0 <= ix_with[i] < nx_act2 and 0 <= iy_with[i] < ny_act2 and 0 <= iz_with[i] < nz_act2:
                E_with_3d[int(ix_with[i]), int(iy_with[i]), int(iz_with[i])] = E_intensity_with[i]
        
        y_slice = ny_act2 // 2
        slice_with = E_with_3d[:, y_slice, :].T
        slice_with_clean = np.where(np.isnan(slice_with) | (slice_with == 0),
                                   slice_with[slice_with > 0].min() if np.any(slice_with > 0) else 0,
                                   slice_with)
        
        im2 = axes_xz[row_idx, 1].imshow(slice_with_clean, aspect='auto', origin='lower',
                                        extent=[0, nx_act2, 0, nz_act2], cmap='jet', interpolation='bilinear')
        axes_xz[row_idx, 1].set_title(f'WITH surf - Corr={corr_len}')
        axes_xz[row_idx, 1].set_xlabel('X'); axes_xz[row_idx, 1].set_ylabel('Z')
        plt.colorbar(im2, ax=axes_xz[row_idx, 1], label='|E|^2')

plt.tight_layout()
plt.savefig(figures_dir / '03_internal_field_xz_slice.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    保存: 03_internal_field_xz_slice.png")

# 2.2 3D连续透明可视化（选择几个相关长）
print("  生成3D连续透明可视化...")
for corr_len in selected_corr:
    corr_dir = base_dir / f'corr_{corr_len}'
    nosurf_dir = corr_dir / 'nosurf'
    
    if not (nosurf_dir / 'IntField-X').exists():
        continue
    
    x_coords, y_coords, z_coords, E_intensity = load_internal_field(
        nosurf_dir / 'IntField-X', nosurf_dir / 'IntField-Y'
    )
    coords = np.column_stack([x_coords, y_coords, z_coords])
    ix, iy, iz, nx_act, ny_act, nz_act = coords_to_indices(coords, nx, ny, 100)
    
    E_3d = np.zeros((nx_act, ny_act, nz_act))
    for i in range(len(ix)):
        if 0 <= ix[i] < nx_act and 0 <= iy[i] < ny_act and 0 <= iz[i] < nz_act:
            E_3d[int(ix[i]), int(iy[i]), int(iz[i])] = E_intensity[i]
    
    # 3D可视化
    fig_3d = plt.figure(figsize=(12, 9), facecolor='white')
    ax = fig_3d.add_subplot(111, projection='3d')
    
    x_coords_3d, y_coords_3d, z_coords_3d = np.meshgrid(
        np.arange(nx_act), np.arange(ny_act), np.arange(nz_act), indexing='ij'
    )
    
    layers_to_show = list(range(0, nz_act, max(1, nz_act//15)))  # 显示约15层
    
    for z_layer in layers_to_show:
        E_layer = E_3d[:, :, z_layer]
        x_flat = x_coords_3d[:, :, z_layer].flatten()
        y_flat = y_coords_3d[:, :, z_layer].flatten()
        E_flat = E_layer.flatten()
        
        mask = E_flat > 0
        if not np.any(mask):
            continue
        
        x_valid = x_flat[mask]
        y_valid = y_flat[mask]
        E_valid = E_flat[mask]
        
        interp_factor = 2
        x_dense = np.linspace(0, nx_act-1, nx_act * interp_factor)
        y_dense = np.linspace(0, ny_act-1, ny_act * interp_factor)
        X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
        
        E_interp = scipy_griddata(
            np.column_stack([x_valid, y_valid]),
            E_valid,
            (X_dense, Y_dense),
            method='cubic',
            fill_value=0
        )
        
        E_min = E_interp[E_interp > 0].min() if np.any(E_interp > 0) else 0
        E_max = E_interp.max()
        if E_max > E_min:
            E_norm = (E_interp - E_min) / (E_max - E_min)
        else:
            E_norm = np.zeros_like(E_interp)
        colors = plt.cm.jet(E_norm)
        
        z_surface = np.full_like(X_dense, z_layer)
        ax.plot_surface(X_dense, Y_dense, z_surface, facecolors=colors,
                       alpha=0.25, shade=False, edgecolor='none', linewidth=0, antialiased=True)
    
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f'3D Internal Field - Corr={corr_len}')
    ax.set_xlim([0, nx_act-1]); ax.set_ylim([0, ny_act-1]); ax.set_zlim([0, nz_act-1])
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'04_internal_field_3d_corr{corr_len}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    保存: 04_internal_field_3d_corr{corr_len}.png")

print("\n" + "="*70)
print("所有可视化完成！")
print("="*70)
print(f"\n结果保存在: {figures_dir}/")

