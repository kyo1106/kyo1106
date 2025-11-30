#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
项目配置文件 - 在移植到其他电脑时需要修改此文件
"""
import os
from pathlib import Path

# ========== ADDA路径配置 ==========
# ADDA可执行文件路径（相对于项目根目录或绝对路径）
# Windows示例: r'.\adda\win64\adda_ocl.exe' 或 r'C:\path\to\adda\win64\adda_ocl.exe'
# Linux示例: r'./adda/linux64/adda' 或 r'/path/to/adda/linux64/adda'
ADDA_EXECUTABLE = r'.\adda\win64\adda_ocl.exe'

# GPU设备编号（如果使用OpenCL GPU加速）
ADDA_GPU_ID = 0

# ========== ADDA输入文件路径 ==========
# 散射网格输入文件路径（相对于项目根目录或绝对路径）
SCAT_GRID_INPUT_FILE = r'adda\examples\papers\2025_surface\scat_grid_dense_1deg.dat'

# ========== 其他配置 ==========
# 如果ADDA不在项目目录下，可以设置ADDA的根目录
# 例如：ADDA_ROOT = r'C:\Program Files\ADDA'
# 如果为None，则使用相对路径
ADDA_ROOT = None

def get_adda_executable():
    """获取ADDA可执行文件的完整路径"""
    if os.path.isabs(ADDA_EXECUTABLE):
        return ADDA_EXECUTABLE
    elif ADDA_ROOT:
        return os.path.join(ADDA_ROOT, ADDA_EXECUTABLE.lstrip('.\\'))
    else:
        # 相对于项目根目录
        project_root = Path(__file__).parent
        return str(project_root / ADDA_EXECUTABLE.lstrip('.\\'))

def get_scat_grid_input():
    """获取散射网格输入文件的完整路径"""
    if os.path.isabs(SCAT_GRID_INPUT_FILE):
        return SCAT_GRID_INPUT_FILE
    elif ADDA_ROOT:
        return os.path.join(ADDA_ROOT, SCAT_GRID_INPUT_FILE.replace('\\', os.sep))
    else:
        # 相对于项目根目录
        project_root = Path(__file__).parent
        return str(project_root / SCAT_GRID_INPUT_FILE.replace('\\', os.sep))

