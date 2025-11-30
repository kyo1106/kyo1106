散射仿真和可视化

## 项目结构

```
adda_project/
├── adda/                    # ADDA源码
├── scripts/                 # 核心脚本
│   ├── shape_generation/    # Shape生成脚本
│   ├── coordinate_transform/ # 坐标转换工具
│   ├── visualization/       # 可视化工具
│   └── simulation/          # 仿真运行脚本
├── figures/                 # 重要图表
│   ├── important/          # 核心结果图表
│   └── archive/             # 归档图表
├── docs/                    # 文档
└── README.md               # 本文件
```

## 核心功能

### 1. Shape生成 (`scripts/shape_generation/`)
- `gen_flat_slab_*.py` - 生成平板slab的shape文件
- `gen_rayleigh_sphere.py` - 生成Rayleigh散射球体
- `gen_shape.py` - 通用shape生成工具

### 2. 坐标转换 (`scripts/coordinate_transform/`)
- `convert_nosurf_to_lab_frame.py` - **核心功能**：将nosurf的beam frame转换为lab frame
  - 这是解决角度坐标系问题的关键工具
  - 使用方法：见脚本内注释

### 3. 可视化 (`scripts/visualization/`)
- `visualize_corrected_comparison.py` - 修正后的对比可视化
- `visualize_3d_detailed.py` - 详细3D可视化

### 4. 仿真运行 (`scripts/simulation/`)
- `run_rayleigh_oblique.py` - 运行Rayleigh散射斜入射仿真
- `run_rayleigh_far_surf.py` - 运行远距离基板的Rayleigh散射仿真

## 重要发现

### 角度坐标系问题

**关键发现**：`nosurf`和`withsurf`模式使用不同的坐标系：
- **nosurf**: beam frame（相对入射光束）
- **withsurf**: lab frame（绝对坐标系，z轴为基板法线）

**解决方案**：使用`convert_nosurf_to_lab_frame.py`将nosurf结果转换到lab frame后再对比。

详细说明见：`docs/ANGLE_OFFSET_DISCOVERY.md`

## 使用示例

### 生成shape文件
```bash
python scripts/shape_generation/gen_flat_slab_100x30.py
```

### 运行仿真
```bash
python scripts/simulation/run_rayleigh_oblique.py
```

### 坐标转换和可视化
```bash
# 转换nosurf结果到lab frame
python scripts/coordinate_transform/convert_nosurf_to_lab_frame.py

# 可视化对比
python scripts/visualization/visualize_corrected_comparison.py
```

## 文档

- `docs/FINAL_INVESTIGATION_REPORT.md` - 完整调查报告
- `docs/ANGLE_OFFSET_DISCOVERY.md` - 角度偏移发现总结
- `docs/VISUALIZATION_INDEX.md` - 可视化图表索引

## 注意事项

1. **坐标系统一**：对比nosurf和withsurf结果时，必须先转换坐标系
2. **入射角影响**：斜入射时，角度偏移 = 入射方向 - 镜面反射方向
3. **可视化**：始终在lab frame下进行可视化，便于与实验对比

## 许可证

ADDA软件遵循其原始许可证。

