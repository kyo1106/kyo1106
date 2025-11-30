# 核心文件列表

## 配置文件
- `config.py` - 项目根目录配置文件
- `rough_surface_simulation/config.py` - 粗糙表面仿真配置文件

## 粗糙表面仿真核心脚本
- `rough_surface_simulation/generate_rough_surface.py` - 生成粗糙表面和shape文件
- `rough_surface_simulation/run_rough_surface_simulation.py` - 主仿真脚本（运行完整仿真）
- `rough_surface_simulation/complete_simulations.py` - 补全未完成的仿真
- `rough_surface_simulation/visualize_all.py` - 生成所有可视化
- `rough_surface_simulation/check_status.py` - 检查仿真状态
- `rough_surface_simulation/calculate_surface_z.py` - 计算surface_z值（辅助工具）

## 平板验证脚本
- `run_flat_slab_validation.py` - 平板验证仿真（nosurf vs withsurf对比）
- `visualize_internal_field.py` - 内部电场可视化（被run_flat_slab_validation.py调用）

## 文档
- `README_PORT.md` - 项目移植指南
- `rough_surface_simulation/README_PORT.md` - 粗糙表面仿真详细指南

## 数据目录（仿真结果）
- `rough_surface_simulation/corr_*/` - 各相关长的仿真结果
- `rough_surface_simulation/figures/` - 可视化结果
- `flat_slab_validation/` - 平板验证结果

## 使用说明

### 运行粗糙表面仿真：
```bash
# 1. 运行完整仿真
python rough_surface_simulation/run_rough_surface_simulation.py

# 2. 检查状态
python rough_surface_simulation/check_status.py

# 3. 补全未完成的仿真
python rough_surface_simulation/complete_simulations.py

# 4. 生成可视化
python rough_surface_simulation/visualize_all.py
```

### 运行平板验证：
```bash
python run_flat_slab_validation.py
```

