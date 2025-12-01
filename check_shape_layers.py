import glob
import os
import re

print(f"{'File':<40} | {'Total Z':<8} | {'Details (Header Comment)'}")
print("-" * 100)

shape_files = glob.glob("rough_surface_simulation/corr_*/rough_surface_corr*.shape")
shape_files.sort()

for shape_file in shape_files:
    try:
        with open(shape_file, 'r', encoding='utf-8', errors='ignore') as f:
            header_line = f.readline().strip()
            # 尝试解析头部注释，例如: # Rough surface slab 64x64x30 (Air 15 layers + Rough Si 1 layers)
            match = re.search(r'Air (\d+) layers \+ Rough Si (\d+) layers', header_line)
            
            # 读取实际的 dimensions (通常在 shape 文件里不直接写 dimensions，但在注释里有)
            # 或者统计最大 Z 值
            
        print(f"{os.path.basename(shape_file):<40} | {'?':<8} | {header_line}")
    except Exception as e:
        print(f"{os.path.basename(shape_file):<40} | Error: {e}")

