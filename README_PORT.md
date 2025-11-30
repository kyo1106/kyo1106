# 项目移植指南

## 在另一台电脑上使用此项目

### 1. 拷贝项目文件

将整个项目文件夹拷贝到新电脑上。

### 2. 修改配置文件

项目中有两个配置文件，根据你使用的脚本选择修改：

- **项目根目录的 `config.py`**：用于根目录的脚本（如 `run_flat_slab_validation.py`）
- **`rough_surface_simulation/config.py`**：用于粗糙表面仿真脚本

**建议**：两个配置文件内容相同，修改其中一个即可。如果两个都修改，保持内容一致。

#### 2.1 ADDA可执行文件路径

根据你的ADDA安装位置，修改 `ADDA_EXECUTABLE`：

**Windows示例：**
```python
# 如果ADDA在项目目录下
ADDA_EXECUTABLE = r'.\adda\win64\adda_ocl.exe'

# 或者使用绝对路径
ADDA_EXECUTABLE = r'C:\Program Files\ADDA\win64\adda_ocl.exe'
```

**Linux示例：**
```python
# 如果ADDA在项目目录下
ADDA_EXECUTABLE = r'./adda/linux64/adda'

# 或者使用绝对路径
ADDA_EXECUTABLE = r'/usr/local/adda/linux64/adda'
```

#### 2.2 GPU设备编号（可选）

如果使用OpenCL GPU加速，修改 `ADDA_GPU_ID`：
```python
ADDA_GPU_ID = 0  # 使用第一个GPU，改为1使用第二个GPU，以此类推
```

#### 2.3 散射网格输入文件路径

修改 `SCAT_GRID_INPUT_FILE` 指向你的 `scat_grid_dense_1deg.dat` 文件：

```python
# 如果文件在ADDA安装目录下
SCAT_GRID_INPUT_FILE = r'adda\examples\papers\2025_surface\scat_grid_dense_1deg.dat'

# 或者使用绝对路径
SCAT_GRID_INPUT_FILE = r'C:\Program Files\ADDA\examples\papers\2025_surface\scat_grid_dense_1deg.dat'
```

#### 2.4 ADDA根目录（可选）

如果ADDA安装在系统目录，可以设置 `ADDA_ROOT`：
```python
ADDA_ROOT = r'C:\Program Files\ADDA'  # Windows
# 或
ADDA_ROOT = r'/usr/local/adda'  # Linux
```

设置后，`ADDA_EXECUTABLE` 和 `SCAT_GRID_INPUT_FILE` 将相对于此目录解析。

### 3. 检查Python依赖

确保安装了以下Python包：
```bash
pip install numpy matplotlib scipy
```

### 4. 运行项目

#### 粗糙表面仿真：
```bash
# 运行主仿真
python rough_surface_simulation/run_rough_surface_simulation.py

# 补全未完成的仿真
python rough_surface_simulation/complete_simulations.py

# 生成可视化
python rough_surface_simulation/visualize_all.py

# 检查仿真状态
python rough_surface_simulation/check_status.py
```

#### 平板验证仿真：
```bash
python run_flat_slab_validation.py
```

### 5. 常见问题

#### Q: 找不到ADDA可执行文件
A: 检查 `config.py` 中的 `ADDA_EXECUTABLE` 路径是否正确，确保路径使用正确的路径分隔符（Windows用`\`，Linux用`/`）。

#### Q: 找不到散射网格输入文件
A: 检查 `config.py` 中的 `SCAT_GRID_INPUT_FILE` 路径，确保文件存在。如果文件不在ADDA目录下，可以拷贝到项目目录并修改路径。

#### Q: GPU加速不工作
A: 检查 `ADDA_GPU_ID` 是否正确，或者尝试使用CPU版本（将 `adda_ocl.exe` 改为 `adda.exe`）。

#### Q: 导入config模块失败
A: 确保在项目根目录运行脚本，或者将 `config.py` 所在目录添加到Python路径。

### 6. 文件结构

```
项目根目录/
├── config.py                          # 配置文件（需要修改）
├── run_flat_slab_validation.py         # 平板验证脚本
├── rough_surface_simulation/
│   ├── config.py                      # 配置文件（需要修改）
│   ├── generate_rough_surface.py      # 粗糙表面生成
│   ├── run_rough_surface_simulation.py  # 主仿真脚本
│   ├── complete_simulations.py         # 补全未完成仿真
│   ├── visualize_all.py               # 可视化脚本
│   ├── check_status.py                 # 检查仿真状态
│   ├── README_PORT.md                  # 详细移植指南
│   ├── corr_0/                        # 相关长=0的仿真结果
│   │   ├── nosurf/
│   │   └── withsurf/
│   ├── corr_8/                        # 相关长=8的仿真结果
│   │   ├── nosurf/
│   │   └── withsurf/
│   └── figures/                       # 可视化结果
└── flat_slab_validation/              # 平板验证结果
    ├── nosurf/
    └── withsurf/
```

### 7. 注意事项

- 所有路径可以使用相对路径（相对于项目根目录）或绝对路径
- Windows路径使用 `r'...'` 原始字符串避免转义问题
- Linux路径使用正斜杠 `/`
- 确保ADDA可执行文件有执行权限（Linux）
- 如果修改了配置文件，需要重启Python脚本才能生效

