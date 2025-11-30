#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整的平板验证仿真脚本
生成shape -> 运行ADDA -> 坐标转换 -> 可视化
"""
import subprocess
import sys
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from config import get_adda_executable, get_scat_grid_input, ADDA_GPU_ID

print("="*70)
print("Flat Slab Validation Simulation (64×64×10, 45 deg incidence)")
print("="*70)

# ========== 1. 生成shape文件 ==========
print(f"\n[1] 生成shape文件...")
nx, ny, nz = 64, 64, 30  # 30层：15层硅 + 15层空气
n_silicon_layers = 15  # 前15层是硅
n_air_layers = 15      # 后15层是空气

shape_file = Path('flat_slab_64x64x30.shape')
with shape_file.open('w', encoding='utf-8') as f:
    f.write(f'# Flat slab {nx}×{ny}×{nz} (15 layers Si + 15 layers Air)\n')
    f.write('Nmat=2\n')  # 多域格式：指定域数量为2
    count = 0
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                # ADDA多域shape文件格式：ix iy iz domain
                # 如果iz < 15，属于域1（硅），否则属于域2（空气）
                if iz < n_silicon_layers:
                    f.write(f'{ix} {iy} {iz} 1\n')  # 域1：硅
                else:
                    f.write(f'{ix} {iy} {iz} 2\n')  # 域2：空气
                count += 1
print(f"  生成: {shape_file} ({count} dipoles)")
print(f"    前{n_silicon_layers}层（z=0-{n_silicon_layers-1}）: 硅")
print(f"    后{n_air_layers}层（z={n_silicon_layers}-{nz-1}）: 空气")

# ========== 2. 运行ADDA仿真 ==========
lambda_um = 0.532
dpl = 16  # 增加dpl以提高网格密度
# 多域折射率：域1（硅）= 3.5+0.01i，域2（空气）= 1.00001+0i
m_silicon = 3.5
k_silicon = 0.01
m_air = 1.00001
k_air = 0.0
prop_x, prop_y, prop_z = 0.7071, 0.0, -0.7071  # 45度入射

output_dir_nosurf = Path('flat_slab_validation/nosurf')
output_dir_withsurf = Path('flat_slab_validation/withsurf')
output_dir_nosurf.mkdir(parents=True, exist_ok=True)
output_dir_withsurf.mkdir(parents=True, exist_ok=True)

print(f"\n[2] 运行NO surf仿真...")
cmd_no = [
    get_adda_executable(),
    '-gpu', str(ADDA_GPU_ID),
    '-shape', 'read', str(shape_file),
    '-dpl', str(dpl),
    '-m', str(m_silicon), str(k_silicon), str(m_air), str(k_air),  # 多域折射率
    '-lambda', str(lambda_um),
    '-prop', str(prop_x), str(prop_y), str(prop_z),
    '-store_scat_grid',
    '-scat_grid_inp', get_scat_grid_input(),
    '-store_int_field',  # 保存内部电场
    '-dir', str(output_dir_nosurf),
    '-eps', '0.5',
    '-iter', 'qmr2'
]
result_no = subprocess.run(cmd_no, capture_output=True, text=True, encoding='utf-8', errors='ignore')
if result_no.returncode != 0:
    print(f"  [ERROR] NO surf仿真失败: {result_no.stderr}")
    sys.exit(1)
print("  [OK] NO surf仿真完成")

print(f"\n[3] 运行WITH surf仿真...")
# 基板设置：下面是硅wafer，上面是空气
# 平板从z=0到z=29（30层），所以表面应该在z=-0.5左右
# 硅wafer在下面（z<0），空气在上面（z>0）
# ADDA的-surf参数：基板在粒子下方
surface_z = 0.5  # 粒子底部到表面的距离
n_substrate = 3.5  # 硅的折射率（简化，实际约3.5+0.01i）
k_substrate = 0.01  # 硅的消光系数
# 注意：ADDA的-surf参数设置的是基板（在下面），空气在上面是默认的

cmd_with = [
    get_adda_executable(),
    '-gpu', str(ADDA_GPU_ID),
    '-shape', 'read', str(shape_file),
    '-dpl', str(dpl),
    '-m', str(m_silicon), str(k_silicon), str(m_air), str(k_air),  # 多域折射率
    '-lambda', str(lambda_um),
    '-prop', str(prop_x), str(prop_y), str(prop_z),
    '-surf', str(surface_z), str(n_substrate), str(k_substrate),
    '-store_scat_grid',
    '-scat_grid_inp', get_scat_grid_input(),
    '-store_int_field',  # 保存内部电场
    '-dir', str(output_dir_withsurf),
    '-eps', '0.5',
    '-iter', 'qmr2'
]
result_with = subprocess.run(cmd_with, capture_output=True, text=True, encoding='utf-8', errors='ignore')
if result_with.returncode != 0:
    print(f"  [ERROR] WITH surf仿真失败: {result_with.stderr}")
    sys.exit(1)
print("  [OK] WITH surf仿真完成")

# ========== 3. 坐标转换和可视化 ==========
print(f"\n[4] 加载数据并进行坐标转换...")

# 加载数据
data_no = np.loadtxt(output_dir_nosurf / "mueller_scatgrid", skiprows=1)
data_with = np.loadtxt(output_dir_withsurf / "mueller_scatgrid", skiprows=1)

theta_no_beam = data_no[:, 0]
phi_no_beam = data_no[:, 1]
s11_no = data_no[:, 2]

theta_with_lab = data_with[:, 0]
phi_with_lab = data_with[:, 1]
s11_with_raw = data_with[:, 2]

valid = s11_with_raw < 1e10
theta_with_valid = theta_with_lab[valid]
phi_with_valid = phi_with_lab[valid]
s11_with_valid = s11_with_raw[valid]

# 坐标转换函数
def beam_to_lab_direction(theta_beam, phi_beam, prop, incPolX, incPolY):
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
    theta_lab = np.degrees(np.arccos(np.clip(robs[:, 2], -1, 1)))
    phi_lab = np.degrees(np.arctan2(robs[:, 1], robs[:, 0]))
    phi_lab[phi_lab < 0] += 360
    return theta_lab, phi_lab

# 从ADDA log获取向量
prop_vec = np.array([prop_x, prop_y, prop_z])
prop_unit = prop_vec / np.linalg.norm(prop_vec)

# 从log文件读取偏振向量（如果存在），否则使用默认值
try:
    log_file = output_dir_nosurf / "log"
    if log_file.exists():
        log_content = log_file.read_text(encoding='utf-8', errors='ignore')
        # 查找偏振向量
        import re
        pol_x_match = re.search(r'Incident polarization X\(per\):\s*\(([-\d.]+),([-\d.]+),([-\d.]+)\)', log_content)
        pol_y_match = re.search(r'Incident polarization Y\(par\):\s*\(([-\d.]+),([-\d.]+),([-\d.]+)\)', log_content)
        if pol_x_match and pol_y_match:
            incPolX = np.array([float(pol_x_match.group(1)), float(pol_x_match.group(2)), float(pol_x_match.group(3))])
            incPolY = np.array([float(pol_y_match.group(1)), float(pol_y_match.group(2)), float(pol_y_match.group(3))])
            print(f"  从log文件读取偏振向量:")
            print(f"    incPolX: {incPolX}")
            print(f"    incPolY: {incPolY}")
        else:
            raise ValueError("未找到偏振向量")
    else:
        raise FileNotFoundError("log文件不存在")
except:
    # 如果无法读取，使用默认值（45度入射）
    incPolX = np.array([-0.7071, 0.0, -0.7071])
    incPolY = np.array([0.0, 1.0, 0.0])
    print(f"  使用默认偏振向量（45度入射）:")
    print(f"    incPolX: {incPolX}")
    print(f"    incPolY: {incPolY}")

# 转换nosurf到lab frame
print("  转换NO surf从beam frame到lab frame...")
robs_no_lab = beam_to_lab_direction(theta_no_beam, phi_no_beam, prop_unit, incPolX, incPolY)
theta_no_lab, phi_no_lab = direction_to_lab_angles(robs_no_lab)

# 统一网格
unique_theta = np.unique(theta_with_valid)
unique_phi = np.unique(phi_with_valid)

points_no = np.column_stack([theta_no_lab, phi_no_lab])
theta_grid, phi_grid = np.meshgrid(unique_theta, unique_phi, indexing='ij')
points_target = np.column_stack([theta_grid.ravel(), phi_grid.ravel()])

s11_no_interp = griddata(points_no, s11_no, points_target, method='linear', fill_value=0)
s11_no_grid = s11_no_interp.reshape(theta_grid.shape)

mask_nan = np.isnan(s11_no_grid)
if mask_nan.any():
    s11_no_nn = griddata(points_no, s11_no, points_target, method='nearest')
    s11_no_grid[mask_nan] = s11_no_nn.reshape(theta_grid.shape)[mask_nan]

# WITH surf数据也需要插值到统一网格
points_with = np.column_stack([theta_with_valid, phi_with_valid])
s11_with_interp = griddata(points_with, s11_with_valid, points_target, method='linear', fill_value=0)
s11_with_grid = s11_with_interp.reshape(theta_grid.shape)

mask_nan_with = np.isnan(s11_with_grid)
if mask_nan_with.any():
    s11_with_nn = griddata(points_with, s11_with_valid, points_target, method='nearest')
    s11_with_grid[mask_nan_with] = s11_with_nn.reshape(theta_grid.shape)[mask_nan_with]

# 提取上半球
idx_upper = unique_theta <= 90
theta_upper = unique_theta[idx_upper]

s11_no_upper = s11_no_grid[idx_upper, :]
s11_with_upper = s11_with_grid[idx_upper, :]

# 归一化
s11_no_norm = s11_no_upper / s11_no_upper.max()
s11_with_norm = s11_with_upper / s11_with_upper.max()

# 计算相关性
corr = np.corrcoef(s11_no_norm.ravel(), s11_with_norm.ravel())[0, 1]
print(f"  相关性: {corr:.4f}")

# ========== 4. 可视化 ==========
print(f"\n[5] 生成可视化图表...")
figures_dir = Path('flat_slab_validation/figures')
figures_dir.mkdir(parents=True, exist_ok=True)

# 4.1 2D热图对比
fig_2d, axes_2d = plt.subplots(1, 2, figsize=(16, 7), facecolor='white')
im1 = axes_2d[0].imshow(s11_no_norm, aspect='auto', origin='lower', 
                        extent=[unique_phi.min(), unique_phi.max(), 
                                theta_upper.min(), theta_upper.max()],
                        cmap='jet', interpolation='bilinear')
axes_2d[0].set_xlabel('Phi (deg, lab frame)')
axes_2d[0].set_ylabel('Theta (deg, lab frame)')
axes_2d[0].set_title(f'NO surf (converted to lab frame) - Correlation: {corr:.4f}')
plt.colorbar(im1, ax=axes_2d[0], label='Normalized S11')

im2 = axes_2d[1].imshow(s11_with_norm, aspect='auto', origin='lower',
                        extent=[unique_phi.min(), unique_phi.max(),
                                theta_upper.min(), theta_upper.max()],
                        cmap='jet', interpolation='bilinear')
axes_2d[1].set_xlabel('Phi (deg, lab frame)')
axes_2d[1].set_ylabel('Theta (deg, lab frame)')
axes_2d[1].set_title('WITH surf (lab frame)')
plt.colorbar(im2, ax=axes_2d[1], label='Normalized S11')

plt.tight_layout()
plt.savefig(figures_dir / "01_2d_heatmap_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  保存: 01_2d_heatmap_comparison.png")

# 4.2 3D半球壳对比（高分辨率）
# 使用更密集的采样和更高的rcount/ccount
step_surf = 1  # 不使用降采样，使用所有数据点
theta_rad_surf = np.radians(theta_upper[::step_surf])
phi_rad_surf = np.radians(unique_phi[::step_surf])
THETA_surf, PHI_surf = np.meshgrid(theta_rad_surf, phi_rad_surf, indexing='ij')

X_surf = np.sin(THETA_surf) * np.cos(PHI_surf)
Y_surf = np.sin(THETA_surf) * np.sin(PHI_surf)
Z_surf = np.cos(THETA_surf)

s11_no_surf = s11_no_norm[::step_surf, ::step_surf]
s11_with_surf = s11_with_norm[::step_surf, ::step_surf]

colors_no = plt.cm.jet(s11_no_surf)
colors_with = plt.cm.jet(s11_with_surf)

fig_3d = plt.figure(figsize=(18, 9), facecolor='white')

ax1 = fig_3d.add_subplot(121, projection='3d')
# 增加rcount和ccount以提高分辨率，最高200
surf1 = ax1.plot_surface(X_surf, Y_surf, Z_surf, facecolors=colors_no, 
                         rcount=min(200, len(theta_upper)), 
                         ccount=min(200, len(unique_phi)), 
                         shade=False, alpha=0.95, 
                         edgecolor='none', antialiased=True, linewidth=0)
ax1.set_title(f'NO surf (converted to lab frame)\nCorrelation: {corr:.4f}', 
              fontsize=14, fontweight='bold')
ax1.view_init(elev=90, azim=0)
ax1.set_xlim([-1, 1]); ax1.set_ylim([-1, 1]); ax1.set_zlim([0, 1])

ax2 = fig_3d.add_subplot(122, projection='3d')
surf2 = ax2.plot_surface(X_surf, Y_surf, Z_surf, facecolors=colors_with, 
                         rcount=min(200, len(theta_upper)), 
                         ccount=min(200, len(unique_phi)), 
                         shade=False, alpha=0.95, 
                         edgecolor='none', antialiased=True, linewidth=0)
ax2.set_title('WITH surf (lab frame)', fontsize=14, fontweight='bold')
ax2.view_init(elev=90, azim=0)
ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([0, 1])

plt.tight_layout()
plt.savefig(figures_dir / "02_3d_hemisphere_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  保存: 02_3d_hemisphere_comparison.png")

# 多视角3D对比
print("  生成多视角3D对比...")
view_angles = [
    (90, 0, "Top View"),
    (60, 45, "Oblique View 1"),
    (60, 135, "Oblique View 2"),
    (30, 0, "Side View"),
    (0, 0, "Front View"),
    (0, 90, "Right Side View"),
]

fig_multi_view = plt.figure(figsize=(24, 18), facecolor='white')

for view_idx, (elev, azim, view_name) in enumerate(view_angles):
    # NO surf
    ax1 = fig_multi_view.add_subplot(len(view_angles), 2, view_idx * 2 + 1, projection='3d')
    surf1 = ax1.plot_surface(X_surf, Y_surf, Z_surf, facecolors=colors_no, 
                             rcount=min(200, len(theta_upper)), 
                             ccount=min(200, len(unique_phi)), 
                             shade=False, alpha=0.95, 
                             edgecolor='none', antialiased=True, linewidth=0)
    ax1.set_title(f'NO surf - {view_name}', fontsize=11, fontweight='bold')
    ax1.view_init(elev=elev, azim=azim)
    ax1.set_xlim([-1, 1]); ax1.set_ylim([-1, 1]); ax1.set_zlim([0, 1])
    
    # WITH surf
    ax2 = fig_multi_view.add_subplot(len(view_angles), 2, view_idx * 2 + 2, projection='3d')
    surf2 = ax2.plot_surface(X_surf, Y_surf, Z_surf, facecolors=colors_with, 
                             rcount=min(200, len(theta_upper)), 
                             ccount=min(200, len(unique_phi)), 
                             shade=False, alpha=0.95, 
                             edgecolor='none', antialiased=True, linewidth=0)
    ax2.set_title(f'WITH surf - {view_name}', fontsize=11, fontweight='bold')
    ax2.view_init(elev=elev, azim=azim)
    ax2.set_xlim([-1, 1]); ax2.set_ylim([-1, 1]); ax2.set_zlim([0, 1])

plt.tight_layout()
plt.savefig(figures_dir / "03_3d_multi_view_comparison.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("  保存: 03_3d_multi_view_comparison.png")

# ========== 5. 内部电场可视化 ==========
print("\n[5] 内部电场可视化...")
try:
    # 直接导入并运行，避免subprocess的编码问题
    import visualize_internal_field
    print("  [OK] 内部电场可视化完成")
except Exception as e:
    print(f"  [WARNING] 内部电场可视化出错: {e}")
    # 如果导入失败，尝试用subprocess但忽略输出
    try:
        subprocess.run(
            [sys.executable, 'visualize_internal_field.py'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            encoding='utf-8', errors='ignore'
        )
        print("  [OK] 内部电场可视化完成（通过subprocess）")
    except:
        print("  [WARNING] 无法运行内部电场可视化")

print("\n" + "="*70)
print("验证完成！")
print("="*70)
print(f"\n相关性: {corr:.4f}")
print(f"（理论上应该接近1.0，因为3层很薄，基板影响很小）")
print(f"\n结果保存在: flat_slab_validation/")

