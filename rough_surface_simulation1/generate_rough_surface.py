#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成粗糙表面的高度图
使用高斯随机场方法，通过相关长控制粗糙度
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def generate_rough_surface(nx, ny, correlation_length, rms_height, seed=None):
    """
    生成粗糙表面高度图
    
    参数:
        nx, ny: 表面尺寸（dipole数量）
        correlation_length: 相关长（dipole单位）
        rms_height: RMS高度（dipole单位）
        seed: 随机种子
    
    返回:
        height_map: (nx, ny) 数组，表面高度
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 如果相关长为0，返回平面（高度为0）
    if correlation_length == 0:
        return np.zeros((nx, ny))
    
    # 生成高斯白噪声
    white_noise = np.random.randn(nx, ny)
    
    # 创建高斯滤波器（相关函数）
    # 相关长越大，表面越平滑
    x = np.arange(nx) - nx // 2
    y = np.arange(ny) - ny // 2
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 高斯相关函数
    sigma = correlation_length / (2 * np.sqrt(2 * np.log(2)))  # FWHM到sigma的转换
    correlation_filter = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    
    # 归一化滤波器
    correlation_filter = correlation_filter / np.sum(correlation_filter)
    
    # 使用FFT卷积
    noise_fft = np.fft.fft2(white_noise)
    filter_fft = np.fft.fft2(np.fft.fftshift(correlation_filter))
    correlated_noise_fft = noise_fft * filter_fft
    correlated_noise = np.real(np.fft.ifft2(correlated_noise_fft))
    
    # 归一化到指定的RMS高度
    current_rms = np.std(correlated_noise)
    if current_rms > 0:
        height_map = correlated_noise * (rms_height / current_rms)
    else:
        height_map = np.zeros_like(correlated_noise)
    
    return height_map

def create_shape_with_rough_surface(nx, ny, nz_air, nz_silicon, height_map, output_file):
    """
    创建带粗糙表面的shape文件
    
    参数:
        nx, ny: x和y方向的尺寸
        nz_air: 空气层数（上层）
        nz_silicon: 硅层数（下层，包含粗糙表面）
        height_map: (nx, ny) 表面高度图（相对于z=0）
        output_file: 输出文件路径
    """
    # 将高度图转换为整数索引（向上取整）
    height_map_int = np.ceil(height_map).astype(int)
    
    # 找到最小高度（可能为负）
    min_height = height_map_int.min()
    
    # 调整高度图，使最小值为0
    height_map_int = height_map_int - min_height
    
    # 计算实际需要的总层数
    max_height = height_map_int.max()
    total_layers = nz_air + max_height + nz_silicon
    
    print(f"  表面高度范围: {min_height} 到 {max_height + min_height}")
    print(f"  调整后高度范围: 0 到 {max_height}")
    print(f"  总层数: {total_layers} (空气{nz_air}层 + 表面{max_height+1}层 + 硅{nz_silicon}层)")
    
    with output_file.open('w', encoding='utf-8') as f:
        f.write(f'# Rough surface slab {nx}×{ny}×{total_layers} (Air {nz_air} layers + Rough Si {nz_silicon} layers)\n')
        f.write('Nmat=2\n')  # 多域：域1=硅，域2=空气
        
        count = 0
        count_si = 0
        count_air = 0
        
        # 遍历所有层
        for iz in range(total_layers):
            for iy in range(ny):
                for ix in range(nx):
                    # 计算当前(x,y)位置的表面高度
                    surface_height = height_map_int[ix, iy]
                    
                    # 计算当前(x,y)位置的实际表面高度（相对于z=0）
                    actual_surface_z = nz_air + surface_height
                    
                    # 上层：空气层（域2，z < nz_air）
                    if iz < nz_air:
                        f.write(f'{ix} {iy} {iz} 2\n')  # 域2：空气
                        count_air += 1
                    # 表面区域：如果当前z在表面高度以下，填充空气（缺口）
                    elif iz < actual_surface_z:
                        f.write(f'{ix} {iy} {iz} 2\n')  # 域2：空气（缺口）
                        count_air += 1
                    # 下层：硅层（域1，z >= 表面高度）
                    else:
                        f.write(f'{ix} {iy} {iz} 1\n')  # 域1：硅
                        count_si += 1
                    count += 1
        
        print(f"  总dipoles: {count}")
        print(f"  硅dipoles: {count_si}")
        print(f"  空气dipoles: {count_air}")
    
    return total_layers

if __name__ == '__main__':
    # 测试生成
    nx, ny = 64, 64
    correlation_lengths = [0, 8, 16, 24, 32, 40, 48, 56, 64]
    rms_height = 5
    
    output_dir = Path('rough_surface_simulation')
    output_dir.mkdir(exist_ok=True)
    
    for corr_len in correlation_lengths:
        print(f"\n生成相关长={corr_len}的粗糙表面...")
        height_map = generate_rough_surface(nx, ny, corr_len, rms_height, seed=42)
        
        # 保存高度图
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(height_map, cmap='terrain', origin='lower')
        ax.set_title(f'Rough Surface Height Map\nCorrelation Length={corr_len}, RMS={rms_height}')
        ax.set_xlabel('X (dipole index)')
        ax.set_ylabel('Y (dipole index)')
        plt.colorbar(im, ax=ax, label='Height (dipoles)')
        plt.tight_layout()
        plt.savefig(output_dir / f'height_map_corr{corr_len}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  保存高度图: height_map_corr{corr_len}.png")
        
        # 统计信息
        print(f"  高度统计: min={height_map.min():.2f}, max={height_map.max():.2f}, RMS={np.std(height_map):.2f}")

