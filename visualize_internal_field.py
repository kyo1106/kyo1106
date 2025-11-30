#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化内部电场
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata as scipy_griddata

print("="*70)
print("Internal Field Visualization")
print("="*70)

output_dir_nosurf = Path('flat_slab_validation/nosurf')
output_dir_withsurf = Path('flat_slab_validation/withsurf')
figures_dir = Path('flat_slab_validation/figures')
figures_dir.mkdir(parents=True, exist_ok=True)

# 参数
nx, ny, nz = 64, 64, 30  # 平板尺寸（30层）

print(f"\n[1] 加载内部电场数据...")

# 检查文件是否存在
int_field_x_no = output_dir_nosurf / "IntField-X"
int_field_y_no = output_dir_nosurf / "IntField-Y"
int_field_x_with = output_dir_withsurf / "IntField-X"
int_field_y_with = output_dir_withsurf / "IntField-Y"

if not all([int_field_x_no.exists(), int_field_y_no.exists(), 
            int_field_x_with.exists(), int_field_y_with.exists()]):
    print("  [ERROR] 内部电场文件不存在，请先运行仿真")
    exit(1)

# 加载内部电场数据
# ADDA的内部电场格式：
# IntField-X: ix iy iz Ex_re Ex_im Ez_re Ez_im (7列)
# IntField-Y: ix iy iz Ey_re Ey_im (5列)
print("  加载NO surf内部电场...")
try:
    data_x_no = np.loadtxt(int_field_x_no, skiprows=1)
    data_y_no = np.loadtxt(int_field_y_no, skiprows=1)
except:
    print(f"  [ERROR] 无法读取内部电场文件")
    print(f"    检查文件: {int_field_x_no}, {int_field_y_no}")
    exit(1)

print(f"  IntField-X形状: {data_x_no.shape}")
print(f"  IntField-Y形状: {data_y_no.shape}")

print("  加载WITH surf内部电场...")
try:
    data_x_with = np.loadtxt(int_field_x_with, skiprows=1)
    data_y_with = np.loadtxt(int_field_y_with, skiprows=1)
except:
    print(f"  [ERROR] 无法读取内部电场文件")
    print(f"    检查文件: {int_field_x_with}, {int_field_y_with}")
    exit(1)

# 解析数据
# ADDA实际格式：x y z |E|^2 Ex.r Ex.i Ey.r Ey.i Ez.r Ez.i (10列)
# 注意：x, y, z是实际坐标（浮点数），不是索引
print(f"  IntField-X列数: {data_x_no.shape[1]}")
print(f"  IntField-Y列数: {data_y_no.shape[1]}")

if data_x_no.shape[1] == 10:
    # 格式：x y z |E|^2 Ex.r Ex.i Ey.r Ey.i Ez.r Ez.i
    x_no = data_x_no[:, 0]
    y_no = data_x_no[:, 1]
    z_no = data_x_no[:, 2]
    Ex_no = data_x_no[:, 4] + 1j * data_x_no[:, 5]  # Ex.r + i*Ex.i
    Ey_no = data_x_no[:, 6] + 1j * data_x_no[:, 7]  # Ey.r + i*Ey.i
    Ez_no = data_x_no[:, 8] + 1j * data_x_no[:, 9]  # Ez.r + i*Ez.i
    
    # 将坐标转换为索引（假设dipole size已知，从shape文件推断）
    # 对于64x64x10的平板，索引范围是0-63, 0-63, 0-9
    # 坐标范围大约是[-size/2, size/2]
    # 简化：直接使用坐标来匹配
    coords_no = np.column_stack([x_no, y_no, z_no])
else:
    print(f"  [ERROR] 未知的IntField-X格式，列数={data_x_no.shape[1]}")
    exit(1)

if data_x_with.shape[1] == 10:
    x_with = data_x_with[:, 0]
    y_with = data_x_with[:, 1]
    z_with = data_x_with[:, 2]
    Ex_with = data_x_with[:, 4] + 1j * data_x_with[:, 5]
    Ey_with = data_x_with[:, 6] + 1j * data_x_with[:, 7]
    Ez_with = data_x_with[:, 8] + 1j * data_x_with[:, 9]
    coords_with = np.column_stack([x_with, y_with, z_with])
else:
    print(f"  [ERROR] 未知的IntField-X格式，列数={data_x_with.shape[1]}")
    exit(1)

# 将坐标转换为索引
# 假设坐标是连续的，找到最小值和最大值来确定索引
# 对于64x64x10的平板，我们需要将坐标映射到[0, nx-1] x [0, ny-1] x [0, nz-1]
def coords_to_indices(coords, nx, ny, nz):
    """将坐标转换为索引"""
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    z_coords = coords[:, 2]
    
    # 找到唯一坐标并排序
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    z_unique = np.unique(z_coords)
    
    # 创建映射
    x_map = {val: idx for idx, val in enumerate(sorted(x_unique))}
    y_map = {val: idx for idx, val in enumerate(sorted(y_unique))}
    z_map = {val: idx for idx, val in enumerate(sorted(z_unique))}
    
    # 转换
    ix = np.array([x_map[x] for x in x_coords])
    iy = np.array([y_map[y] for y in y_coords])
    iz = np.array([z_map[z] for z in z_coords])
    
    return ix, iy, iz

ix_no, iy_no, iz_no = coords_to_indices(coords_no, nx, ny, nz)
ix_with, iy_with, iz_with = coords_to_indices(coords_with, nx, ny, nz)

# 计算电场强度（也可以直接使用文件中的|E|^2，但这里重新计算以确保一致性）
E_intensity_no = np.abs(Ex_no)**2 + np.abs(Ey_no)**2 + np.abs(Ez_no)**2
E_intensity_with = np.abs(Ex_with)**2 + np.abs(Ey_with)**2 + np.abs(Ez_with)**2

# 重塑为3D数组
E_no_3d = np.zeros((nx, ny, nz))
E_with_3d = np.zeros((nx, ny, nz))

print(f"  填充3D数组...")
print(f"    NO surf: {len(ix_no)} 个数据点")
print(f"    WITH surf: {len(ix_with)} 个数据点")

for i in range(len(ix_no)):
    if 0 <= ix_no[i] < nx and 0 <= iy_no[i] < ny and 0 <= iz_no[i] < nz:
        E_no_3d[int(ix_no[i]), int(iy_no[i]), int(iz_no[i])] = E_intensity_no[i]
    else:
        print(f"    [WARNING] NO surf索引超出范围: ({ix_no[i]}, {iy_no[i]}, {iz_no[i]})")

for i in range(len(ix_with)):
    if 0 <= ix_with[i] < nx and 0 <= iy_with[i] < ny and 0 <= iz_with[i] < nz:
        E_with_3d[int(ix_with[i]), int(iy_with[i]), int(iz_with[i])] = E_intensity_with[i]
    else:
        print(f"    [WARNING] WITH surf索引超出范围: ({ix_with[i]}, {iy_with[i]}, {iz_with[i]})")

print(f"    NO surf 3D数组非零元素: {np.count_nonzero(E_no_3d)}")
print(f"    WITH surf 3D数组非零元素: {np.count_nonzero(E_with_3d)}")
print(f"    NO surf最大值: {E_no_3d.max():.6e}")
print(f"    WITH surf最大值: {E_with_3d.max():.6e}")

print(f"\n[2] 生成内部电场可视化...")

# 2.1 x-z切面（y=ny/2）
y_slice = ny // 2
print(f"  生成x-z切面（y={y_slice}）...")

fig_xz, axes_xz = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')

# 使用线性刻度显示（不用log）
slice_no = E_no_3d[:, y_slice, :].T
slice_with = E_with_3d[:, y_slice, :].T

# 处理空数据：将NaN和0替换为最小值
slice_no_clean = np.where(np.isnan(slice_no) | (slice_no == 0), 
                          slice_no[slice_no > 0].min() if np.any(slice_no > 0) else 0, 
                          slice_no)
slice_with_clean = np.where(np.isnan(slice_with) | (slice_with == 0),
                            slice_with[slice_with > 0].min() if np.any(slice_with > 0) else 0,
                            slice_with)

vmin_no = slice_no_clean[slice_no_clean > 0].min() if np.any(slice_no_clean > 0) else 0
vmax_no = slice_no_clean.max()
vmin_with = slice_with_clean[slice_with_clean > 0].min() if np.any(slice_with_clean > 0) else 0
vmax_with = slice_with_clean.max()

im1 = axes_xz[0].imshow(slice_no_clean, aspect='auto', origin='lower',
                        extent=[0, nx, 0, nz], cmap='jet', 
                        vmin=vmin_no, vmax=vmax_no, interpolation='bilinear')
axes_xz[0].set_xlabel('X (dipole index)')
axes_xz[0].set_ylabel('Z (dipole index)')
axes_xz[0].set_title(f'NO surf - X-Z slice (y={y_slice})\nMax |E|^2 = {slice_no.max():.4e}')
plt.colorbar(im1, ax=axes_xz[0], label='|E|^2')

im2 = axes_xz[1].imshow(slice_with_clean, aspect='auto', origin='lower',
                        extent=[0, nx, 0, nz], cmap='jet',
                        vmin=vmin_with, vmax=vmax_with, interpolation='bilinear')
axes_xz[1].set_xlabel('X (dipole index)')
axes_xz[1].set_ylabel('Z (dipole index)')
axes_xz[1].set_title(f'WITH surf - X-Z slice (y={y_slice})\nMax |E|^2 = {slice_with.max():.4e}')
plt.colorbar(im2, ax=axes_xz[1], label='|E|^2')

plt.tight_layout()
plt.savefig(figures_dir / "04_internal_field_xz_slice.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    保存: 04_internal_field_xz_slice.png")

# 2.2 y-z切面（x=nx/2）
x_slice = nx // 2
print(f"  生成y-z切面（x={x_slice}）...")

fig_yz, axes_yz = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')

slice_no = E_no_3d[x_slice, :, :].T
slice_with = E_with_3d[x_slice, :, :].T

# 处理空数据
slice_no_clean = np.where(np.isnan(slice_no) | (slice_no == 0),
                          slice_no[slice_no > 0].min() if np.any(slice_no > 0) else 0,
                          slice_no)
slice_with_clean = np.where(np.isnan(slice_with) | (slice_with == 0),
                            slice_with[slice_with > 0].min() if np.any(slice_with > 0) else 0,
                            slice_with)

vmin_no = slice_no_clean[slice_no_clean > 0].min() if np.any(slice_no_clean > 0) else 0
vmax_no = slice_no_clean.max()
vmin_with = slice_with_clean[slice_with_clean > 0].min() if np.any(slice_with_clean > 0) else 0
vmax_with = slice_with_clean.max()

im1 = axes_yz[0].imshow(slice_no_clean, aspect='auto', origin='lower',
                        extent=[0, ny, 0, nz], cmap='jet',
                        vmin=vmin_no, vmax=vmax_no, interpolation='bilinear')
axes_yz[0].set_xlabel('Y (dipole index)')
axes_yz[0].set_ylabel('Z (dipole index)')
axes_yz[0].set_title(f'NO surf - Y-Z slice (x={x_slice})\nMax |E|^2 = {slice_no.max():.4e}')
plt.colorbar(im1, ax=axes_yz[0], label='|E|^2')

im2 = axes_yz[1].imshow(slice_with_clean, aspect='auto', origin='lower',
                        extent=[0, ny, 0, nz], cmap='jet',
                        vmin=vmin_with, vmax=vmax_with, interpolation='bilinear')
axes_yz[1].set_xlabel('Y (dipole index)')
axes_yz[1].set_ylabel('Z (dipole index)')
axes_yz[1].set_title(f'WITH surf - Y-Z slice (x={x_slice})\nMax |E|^2 = {slice_with.max():.4e}')
plt.colorbar(im2, ax=axes_yz[1], label='|E|^2')

plt.tight_layout()
plt.savefig(figures_dir / "05_internal_field_yz_slice.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    保存: 05_internal_field_yz_slice.png")

# 2.3 x-y切面（z=nz/2，中间层，在硅和空气的交界处）
z_slice = 15  # 第15层，硅和空气的交界处
print(f"  生成x-y切面（z={z_slice}，中间层）...")

fig_xy, axes_xy = plt.subplots(1, 2, figsize=(16, 8), facecolor='white')

slice_no = E_no_3d[:, :, z_slice]
slice_with = E_with_3d[:, :, z_slice]

# 处理空数据
slice_no_clean = np.where(np.isnan(slice_no) | (slice_no == 0),
                          slice_no[slice_no > 0].min() if np.any(slice_no > 0) else 0,
                          slice_no)
slice_with_clean = np.where(np.isnan(slice_with) | (slice_with == 0),
                            slice_with[slice_with > 0].min() if np.any(slice_with > 0) else 0,
                            slice_with)

vmin_no = slice_no_clean[slice_no_clean > 0].min() if np.any(slice_no_clean > 0) else 0
vmax_no = slice_no_clean.max()
vmin_with = slice_with_clean[slice_with_clean > 0].min() if np.any(slice_with_clean > 0) else 0
vmax_with = slice_with_clean.max()

im1 = axes_xy[0].imshow(slice_no_clean, aspect='auto', origin='lower',
                        extent=[0, nx, 0, ny], cmap='jet',
                        vmin=vmin_no, vmax=vmax_no, interpolation='bilinear')
axes_xy[0].set_xlabel('X (dipole index)')
axes_xy[0].set_ylabel('Y (dipole index)')
axes_xy[0].set_title(f'NO surf - X-Y slice (z={z_slice}, middle layer)\nMax |E|^2 = {slice_no.max():.4e}')
plt.colorbar(im1, ax=axes_xy[0], label='|E|^2')

im2 = axes_xy[1].imshow(slice_with_clean, aspect='auto', origin='lower',
                        extent=[0, nx, 0, ny], cmap='jet',
                        vmin=vmin_with, vmax=vmax_with, interpolation='bilinear')
axes_xy[1].set_xlabel('X (dipole index)')
axes_xy[1].set_ylabel('Y (dipole index)')
axes_xy[1].set_title(f'WITH surf - X-Y slice (z={z_slice}, middle layer)\nMax |E|^2 = {slice_with.max():.4e}')
plt.colorbar(im2, ax=axes_xy[1], label='|E|^2')

plt.tight_layout()
plt.savefig(figures_dir / "06_internal_field_xy_slice.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    保存: 06_internal_field_xy_slice.png")

# 2.4 3D可视化（连续透明表面）
print(f"  生成3D连续透明可视化...")

from scipy.interpolate import griddata as scipy_griddata

# 创建坐标网格
x_coords, y_coords, z_coords = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')

# 显示所有层，使用插值补间创建平滑的连续表面
layers_to_show = list(range(nz))  # 显示所有30层

print(f"    显示所有 {len(layers_to_show)} 层，使用插值补间")

fig_3d = plt.figure(figsize=(18, 9), facecolor='white')

# NO surf
ax1 = fig_3d.add_subplot(121, projection='3d')

# 对每一层创建插值表面
for layer_idx, z_layer in enumerate(layers_to_show):
    # 获取当前层的电场数据
    E_layer = E_no_3d[:, :, z_layer]
    
    # 创建x-y网格用于插值
    x_flat = x_coords[:, :, z_layer].flatten()
    y_flat = y_coords[:, :, z_layer].flatten()
    E_flat = E_layer.flatten()
    
    # 只显示非零数据
    mask = E_flat > 0
    if not np.any(mask):
        continue
    
    x_valid = x_flat[mask]
    y_valid = y_flat[mask]
    E_valid = E_flat[mask]
    
    # 创建更密集的网格用于插值（补间，提高平滑度）
    interp_factor = 2  # 插值因子，增加网格密度
    x_dense = np.linspace(0, nx-1, nx * interp_factor)
    y_dense = np.linspace(0, ny-1, ny * interp_factor)
    X_dense, Y_dense = np.meshgrid(x_dense, y_dense)
    
    # 使用cubic插值创建平滑表面
    E_interp = scipy_griddata(
        np.column_stack([x_valid, y_valid]),
        E_valid,
        (X_dense, Y_dense),
        method='cubic',
        fill_value=0
    )
    
    # 归一化颜色（基于当前层的范围）
    E_min = E_interp[E_interp > 0].min() if np.any(E_interp > 0) else 0
    E_max = E_interp.max()
    if E_max > E_min:
        E_norm = (E_interp - E_min) / (E_max - E_min)
    else:
        E_norm = np.zeros_like(E_interp)
    colors = plt.cm.jet(E_norm)
    
    # 绘制透明表面（alpha值根据层数调整，让所有层都可见）
    z_surface = np.full_like(X_dense, z_layer)
    ax1.plot_surface(X_dense, Y_dense, z_surface, facecolors=colors, 
                    alpha=0.25, shade=False, edgecolor='none', linewidth=0, antialiased=True)

ax1.set_xlabel('X (dipole index)')
ax1.set_ylabel('Y (dipole index)')
ax1.set_zlabel('Z (dipole index, 30 layers)')
ax1.set_title(f'NO surf - 3D Internal Field (Continuous Transparent)')
ax1.set_xlim([0, nx-1])
ax1.set_ylim([0, ny-1])
ax1.set_zlim([0, nz-1])
ax1.view_init(elev=30, azim=45)

# WITH surf
ax2 = fig_3d.add_subplot(122, projection='3d')

for layer_idx, z_layer in enumerate(layers_to_show):
    E_layer = E_with_3d[:, :, z_layer]
    
    x_flat = x_coords[:, :, z_layer].flatten()
    y_flat = y_coords[:, :, z_layer].flatten()
    E_flat = E_layer.flatten()
    
    mask = E_flat > 0
    if not np.any(mask):
        continue
    
    x_valid = x_flat[mask]
    y_valid = y_flat[mask]
    E_valid = E_flat[mask]
    
    interp_factor = 2
    x_dense = np.linspace(0, nx-1, nx * interp_factor)
    y_dense = np.linspace(0, ny-1, ny * interp_factor)
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
    ax2.plot_surface(X_dense, Y_dense, z_surface, facecolors=colors, 
                    alpha=0.25, shade=False, edgecolor='none', linewidth=0, antialiased=True)

ax2.set_xlabel('X (dipole index)')
ax2.set_ylabel('Y (dipole index)')
ax2.set_zlabel('Z (dipole index, 30 layers)')
ax2.set_title(f'WITH surf - 3D Internal Field (Continuous Transparent)')
ax2.set_xlim([0, nx-1])
ax2.set_ylim([0, ny-1])
ax2.set_zlim([0, nz-1])
ax2.view_init(elev=30, azim=45)

plt.tight_layout()
plt.savefig(figures_dir / "07_internal_field_3d.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("    保存: 07_internal_field_3d.png")

print("\n" + "="*70)
print("内部电场可视化完成！")
print("="*70)
print(f"\n结果保存在: {figures_dir}/")

