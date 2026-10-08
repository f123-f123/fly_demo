# 环境依赖与安装

本文以本机已运行的 Windows 11 x64 环境为基准。所有 shell 命令均使用 PowerShell 7，
在项目根目录执行。版本来自本机安装记录；并不表示其他版本一定不兼容。

## 依赖文件

| 文件 | 用途 |
| --- | --- |
| `requirements.txt` | Python 基础运行、离线实验和单元测试依赖，包含固定版本的间接依赖；可用于 CPU 模式。 |
| `requirements-cuda.txt` | 引用基础依赖，并安装 CuPy 与 CUDA 12.9 运行组件；用于现有 GPU 启动脚本。 |
| `environment.yml` | 创建名为 `flybrain` 的 Conda GPU 环境，指定 Python、pip，并安装上述 GPU 依赖。 |
| `package.json` / `package-lock.json` | 浏览器端 Three.js 依赖及其安装锁定信息；使用 `npm ci` 安装。 |

这些文件都应提交到 GitHub。运行时下载的数据、Conda 环境和 `node_modules/` 无需提交。
Python 清单锁定运行包版本；Conda 的系统运行库由求解器安装，未逐个锁定构建号，
因此这不是包含操作系统、驱动及所有二进制哈希的完整环境镜像。

## 系统与工具

| 项目 | 已核实版本或要求 |
| --- | --- |
| 操作系统 | Windows 11 x64；本项目的 PowerShell 启动脚本以 Windows 为目标。 |
| Shell | PowerShell 7，用于安装和运行 `.ps1` 脚本。 |
| 环境管理 | Miniconda 或 Anaconda，终端中可执行 `conda`。 |
| Python | CPython 3.11.16，64 位。 |
| pip | 26.2.1。 |
| Node.js / npm | 本机为 Node.js 26.10.0、npm 12.1.0，仅用于安装前端静态依赖。 |
| 浏览器 | 支持 WebGL2、ES modules 和 WebSocket 的浏览器，如启用硬件加速的 Edge/Chrome。 |
| GPU | CUDA 模式需要 NVIDIA GPU；本机为 GeForce RTX 5060 Laptop GPU。 |
| NVIDIA 驱动 | 本机为 610.88；其他机器需安装支持该 GPU 和 CUDA 12.9 的驱动。610.88 是验证版本，不是推导出的最低版本。 |

系统软件通过 [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)、
[PowerShell](https://learn.microsoft.com/powershell/scripting/install/installing-powershell-on-windows)、
[Node.js](https://nodejs.org/en/download) 和 [NVIDIA](https://www.nvidia.com/Download/index.aspx) 安装。
Git 用于克隆仓库和版本管理，不参与模拟计算。

## Python 与 CUDA 包

主要 Python 包：

| 包 | 固定版本 | 用途 |
| --- | --- | --- |
| FlyBrain | 0.1.0 | 神经网络模拟、连接组数据下载、线性读出工具。 |
| NumPy | 2.4.6 | 感觉编码、世界状态、统计和实验计算。 |
| SciPy | 1.17.1 | 稀疏连接矩阵、权重干预和读出计算。 |
| Numba / llvmlite | 0.67.0 / 0.49.0 | FlyBrain CPU 计算支持；导入 FlyBrain 也需要这些包。 |
| aiohttp | 3.14.3 | HTTP、WebSocket 服务及端到端探针。 |
| CuPy CUDA 12 | 14.2.0 | GPU 数组、稀疏矩阵计算与随机数。 |

FlyBrain 使用 [PyPI 的 `flybrain==0.1.0`](https://pypi.org/project/flybrain/0.1.0/)，
上游源码为 [alextitonis/fly.ai](https://github.com/alextitonis/fly.ai)。本机包没有指向本地源码的 editable 安装记录。

GPU 清单中的 `cupy-cuda12x[ctk]` 会安装 NVIDIA 提供的运行组件，包括 cuBLAS、
cuFFT、cuRAND、cuSOLVER、cuSPARSE、NVRTC、CUDA Runtime 与 nvJitLink，
各组件版本均在 `requirements-cuda.txt` 中列明。本机实际加载的 CUDA Runtime/NVRTC 为 12.9。
这些包不安装显卡驱动；使用这一 wheel 安装路线时，无需另装系统级完整 CUDA Toolkit 或 NVCC。
参见 [CuPy 官方安装说明](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi)。

不要在同一个环境再混装 `cupy`、`cupy-cuda13x` 或另一套 Conda CuPy。
`nvidia-smi` 显示的 CUDA 版本是驱动支持能力，实际使用的 Runtime 以 `cupy.show_config()` 为准。

本项目测试使用 Python 标准库 `unittest`，无需额外安装 pytest。
当前运行和实验无需 PyTorch、TensorFlow、Jupyter、pandas 或 pyarrow。
只有选择从上游原始连接组重新构建数据时，才需要 FlyBrain 的 `[build]` 扩展及原始数据；该路线不属于本安装清单。

## 从零安装 GPU 环境

先安装上面的系统工具和 NVIDIA 驱动。进入克隆后的项目目录，再执行：

```powershell
# 在含有 environment.yml 的项目根目录执行；环境 flybrain 尚未存在时使用。
conda env create -f environment.yml
npm ci

# 首次下载连接组数据；已有数据时直接复用。
conda run -n flybrain python -c "from flybrain.data import ensure_data; print(ensure_data())"

# 检查 Python 依赖，再验证 GPU 和模型计算。
conda run -n flybrain python -m pip check
conda run -n flybrain python -c "import cupy as cp; cp.show_config(); print('GPU sum:', cp.arange(10).sum().item())"
conda run -n flybrain python -c "from flybrain import FlyBrain; b=FlyBrain(device='cuda', sensory_input=False); print('neurons:', b.n, 'spikes:', len(b.step()))"
conda run -n flybrain python -m unittest discover -s tests -v

.\start.ps1
```

`npm ci` 根据已有 lock 文件重建 `node_modules/`。启动后访问 <http://127.0.0.1:8765>。
`start.ps1` 和 `calibrate.ps1` 都明确选择 `--device cuda`。
新机器不要求使用原作者的 `E:\neuro\fly_demo` 或 `D:\miniconda` 路径。

已有 Python 3.11 的 `flybrain` 环境时，可直接同步依赖：

```powershell
conda run -n flybrain python -m pip install -r requirements-cuda.txt
npm ci
conda run -n flybrain python -m pip check
```

此命令会将清单中的包调整为固定版本。需要保留其他项目环境时，先用
`conda env create -f environment.yml -n flybrain-demo` 创建独立环境，之后命令中的环境名也改为
`flybrain-demo`，并直接用 `python -m app.server --device cuda` 启动模块。

## 没有 NVIDIA GPU 的 CPU 路线

在尚未创建 `flybrain` 环境的机器上执行以下命令；这与 GPU 路线二选一：

```powershell
conda create -n flybrain python=3.11.16 pip=26.2.1 -y
conda run -n flybrain python -m pip install -r requirements.txt
npm ci
conda run -n flybrain python -m app.server --device cpu
```

CPU 模式仍需相同模型数据，首次启动会自动下载。不要用固定选择 CUDA 的 `start.ps1`。
GPU 实验文档中的 `--device cuda` 需改成 `--device cpu`；CPU 性能和噪声序列与 GPU 不同，
不能期待实时速度或逐 spike 一致的实验结果。

## 必需的模型数据

FlyBrain 包本身不含连接组。首次构造模型时，会从上游
[brain-v1 release](https://github.com/alextitonis/fly.ai/releases/tag/brain-v1)
下载约 260 MB 的两个文件，默认存到 `$env:USERPROFILE\fly-data`：

| 文件 | SHA-256 |
| --- | --- |
| `brain.npz` | `cc9bd1ecd00bd703a6fa648bc6ad145c93c7c1ee53debdcc9ce0d1f4305e6aca` |
| `weights.npz` | `c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c` |

需要自定义路径时，在启动 Python 前设置环境变量，例如：

```powershell
$env:FLY_DATA = 'E:\neuro\fly-data'
conda run -n flybrain python -c "from flybrain.data import ensure_data; print(ensure_data())"
```

该设置只影响当前 PowerShell 会话及其子进程。离线机器需提前复制这两个文件到同一目录，
并自行准备依赖包；不能只复制本项目源码。
FlyBrain 会校验新下载文件，但会直接复用已存在的文件。可额外检查本机数据：

```powershell
@'
import hashlib
from flybrain.data import DATA, FILES
for name, expected in FILES.items():
    with (DATA / name).open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert actual == expected, f'Checksum mismatch: {name}'
    print(name, actual)
'@ | conda run --no-capture-output -n flybrain python -
```

`config/decoder.json`、`config/neural_routes.json` 和实验 JSON 由本项目随源码提供。
依赖更新不会自动重新校准 Decoder，也不应覆盖历史实验的 baseline manifest。

## 常见环境问题

- `conda` 或 `npm` 找不到：安装对应工具后重开 PowerShell，确认命令已加入 PATH。
- CUDA 启动失败：先检查 `nvidia-smi`，再运行上述 CuPy 和 FlyBrain 验证命令，区分驱动、GPU 包和模型数据问题。
- 仅有 `CUDA path could not be detected` 警告：本机 wheel 路线也会出现该警告；以实际 GPU 运算是否成功为准，不要随意填写不存在的 `CUDA_PATH`。
- `Three.js is missing`：在项目根目录运行 `npm ci`；Three.js 由本地服务器提供，不需要浏览器访问 CDN。
- 数据下载失败：检查能否访问 GitHub release，或将校验通过的数据放入 `FLY_DATA` 指定目录。

本清单依据已有可运行环境编写。版本锁定与当前环境检查不能代替一台全新机器上的完整安装验证。
