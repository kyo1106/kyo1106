#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查仿真状态"""
from pathlib import Path

base_dir = Path('rough_surface_simulation')
correlation_lengths = list(range(0, 33, 4))  # 0, 4, 8, ..., 32

print("="*70)
print("检查withsurf仿真状态")
print("="*70)

for corr_len in correlation_lengths:
    corr_dir = base_dir / f'corr_{corr_len}'
    withsurf_dir = corr_dir / 'withsurf'
    log_file = withsurf_dir / 'log'
    mueller_file = withsurf_dir / 'mueller_scatgrid'
    int_file = withsurf_dir / 'IntField-X'
    
    has_mueller = mueller_file.exists()
    has_int = int_file.exists()
    
    status = []
    if has_mueller and has_int:
        status.append("完成")
    elif log_file.exists():
        log_content = log_file.read_text(encoding='utf-8', errors='ignore')
        if "Total wall time" in log_content:
            status.append("已完成")
        elif "ERROR" in log_content:
            error_line = [l for l in log_content.split('\n') if 'ERROR' in l]
            if error_line:
                status.append(f"错误: {error_line[0][:60]}")
        elif "RE_" in log_content:
            last_lines = log_content.split('\n')[-5:]
            re_line = [l for l in last_lines if 'RE_' in l]
            if re_line:
                status.append(f"运行中: {re_line[-1].strip()[:50]}")
        else:
            status.append("状态未知")
    else:
        status.append("未开始")
    
    print(f"corr_{corr_len:2d}: {', '.join(status)}")

