#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试配置和导入"""
import sys
import traceback

print("="*70)
print("测试配置")
print("="*70)
sys.stdout.flush()

# 测试1: 导入模块
print("\n[1] 测试导入模块...")
sys.stdout.flush()
try:
    from generate_rough_surface import generate_rough_surface, create_shape_with_rough_surface
    print("  ✓ generate_rough_surface 导入成功")
    sys.stdout.flush()
except Exception as e:
    print(f"  ✗ generate_rough_surface 导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    from config import get_adda_executable, get_scat_grid_input, ADDA_GPU_ID
    print("  ✓ config 导入成功")
    sys.stdout.flush()
except Exception as e:
    print(f"  ✗ config 导入失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试2: 生成表面
print("\n[2] 测试生成256x256表面...")
sys.stdout.flush()
try:
    import numpy as np
    height_map = generate_rough_surface(256, 256, 4, 5, seed=42)
    print(f"  ✓ 表面生成成功: shape={height_map.shape}, RMS={np.std(height_map):.2f}")
    sys.stdout.flush()
except Exception as e:
    print(f"  ✗ 表面生成失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# 测试3: 检查ADDA路径
print("\n[3] 测试ADDA路径...")
sys.stdout.flush()
try:
    import os
    adda_exe = get_adda_executable()
    print(f"  ADDA路径: {adda_exe}")
    sys.stdout.flush()
    if os.path.exists(adda_exe):
        print(f"  ✓ ADDA可执行文件存在")
        sys.stdout.flush()
    else:
        print(f"  ✗ ADDA可执行文件不存在: {adda_exe}")
        sys.stdout.flush()
except Exception as e:
    print(f"  ✗ ADDA路径获取失败: {e}")
    traceback.print_exc()

print("\n" + "="*70)
print("配置测试完成！")
print("="*70)
sys.stdout.flush()

