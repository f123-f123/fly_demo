# FlyWorld Lab · FlyBrain 闭环生态箱

这个项目使用 `flybrain 0.1.0` 模拟器，把三维虚拟生态箱产生的视网膜几何与气味场注入
`LC10a`、`LC4/LPLC2` 和已验证存在的食物气味 ORN，再用 FlyBrain spike 活动控制程序化果蝇。
系统只提供 FlyBrain 闭环模式，不包含 Direct Controller 或神经消融对照。

## 信号链

```text
三维生态箱 → 视网膜几何 / 双触角气味采样 → LC10a / LC4 / LPLC2 / ORN → FlyBrain
          → DNa02 / DNp01 / ORN spike → Decoder → 果蝇运动 → 三维生态箱
```

- `turn`：来自左右 `DNa02` firing-rate 差和左右食物气味 ORN spike 差。
- `action`：来自 `DNp01`，触发急停并进入逃逸动作。
- `speed`：Decoder 输出固定基准速度，由 DNp01 action 制动；世界层另有行为状态速度倍率及碰撞恢复减速，均为非神经策略。
- `altitude`：起飞、悬停、降落和逃逸由明确标注的高度保持策略负责；候选飞行 DN 尚未验证，绝不伪装为神经输出。
- Decoder 的函数输入只有 FlyBrain firing rate 和时间步，不读取目标坐标、目标可见性或预期转向。
- 神经模拟为 50 Hz；使用 260 ms 滚动窗口；控制和面板更新为 10 Hz。

## 三维闭环

- 后端是世界状态的唯一权威源，维护 `x/y/z`、三轴速度、yaw、pitch、飞行模式和碰撞。
- 每个 20 ms 步长计算目标水平角、俯仰角、遮挡、视觉角尺寸、`dθ/dt`、接近速度和 TTC。
- 水果与花朵产生受风向、距离、横风偏移和噪声影响的三维气味羽流；左右触角在各自空间位置采样。
- 重置会生成新的确定性随机布局：腐果簇、花朵簇、不同高度的枝叶/枝干/石块、不平整地面和三个局部风区。
- Three.js 和后端共享枝干胶囊体端点、薄叶三角面及球形石块；同一几何用于渲染、遮挡与碰撞。植物从地面连接主干、分枝和叶片。纹理、地表小落叶/苔草和花梗仅为装饰，不参与碰撞；尚非完整生态物理模拟。
- 感觉状态机包含 `explore → odor_tracking → visual_lock → approach → landing`；只有 Decoder 的 DNp01 action 可进入 `escape`，原始 looming 不再绕过神经回路触发动作。概率转换和最短驻留时间减少频繁切换，并非完整阈值迟滞模型。
- 逃逸方向为非神经辅助策略：根据可见威胁的世界垂直方向选择远离方向，结合地面/箱顶余量；无可见威胁的手动 action 仅制动。取消固定向上速度脉冲和碰撞后无条件上抬。该策略不代表已验证的果蝇飞行动力学。
- 辅助逃逸使用有限加速度（最大 10 场景单位/s²）与对称垂直限速（3.0 场景单位/s），并非瞬时赋予向上速度。远处仅视觉锁定时继续探索换层，进入逼近阶段才对齐食物表面高度。
- 垂直加速度、俯仰、悬停漂移与阵风使用有时间相关性的随机过程；探索高度连续积分变化，不再定时随机抽取高度。气味浓度时间变化不再用于猜测上下方向；视觉逼近高度经过身体 pitch/bearing 坐标反变换。
- DNa02/ORN 的 Decoder turn 经过转向 saccade、惯性和侧滑形成运动；随机 yaw 微扰单独记录为策略项，不与神经输出混淆。
- 水果、花朵、枝叶、石块和逼近威胁在重置后自主运行；按钮只用于神经链路诊断。
- JSONL 逐帧记录三维轨迹、气味浓度、双侧输入、spike、Decoder 输出和觅食/降落/避碰统计。
- `motion_policy.height_policy / escape_trigger / escape_direction_policy` 单独记录高度策略、神经逃逸触发与辅助方向。避碰率仅是场景无接触事件统计，不能解释为纯神经贡献或生物学真实性证据。

## 神经通路审计

当前安装数据中已确认 `ORN_DM1` 与 `ORN_VA2` 双侧 population 存在，并通过开环批量刺激确认其自身及上行 PN 响应。
本地测试未观察到可稳定用于嗅觉侧向控制的 DNa02 或候选 DNg 输出，因此当前气味偏转严格使用已验证 ORN 的实际 spike 双侧差。
`DNg01/02/03/05/06` 只记录为候选飞行通路，不承担垂直控制。

重新运行审计：

```powershell
conda run -n flybrain python scripts\audit_neural_routes.py --device cuda --batch 8
```

结果写入 `config/neural_routes.json`。

## 权重干预实验

可复现基线已经冻结为机器可读 manifest，包含原始权重与 Decoder 校验和、核心源码哈希、LC10a/looming Baseline、全脑 spike、随机种子、RTF 和实际测试结果。它从已保存的原始 Baseline 条件回溯生成，并保留这一时间边界：

```powershell
conda run -n flybrain python scripts\freeze_baseline.py
```

基线文件为 `config/experiments/baseline_manifest.json`。

当前已完成 `LC10a → DNa02` 和 `LC4/LPLC2 → DNp01` 的 `Baseline / Sham / 0× / 0.5× / 2×` 五条件实验。干预只作用于进程内的感觉 population 出站权重，不改写原始 `weights.npz`。24 次/侧的结果显示两个通路均有单调神经剂量效应，Sham 与 Baseline 等效，2× 条件没有全脑饱和。

```powershell
conda run -n flybrain python scripts\run_weight_intervention.py --device cuda --batch 8 --seeds 1101,2203,3307
conda run -n flybrain python scripts\run_escape_weight_intervention.py --device cuda --batch 8 --seeds 1101,2203,3307
```

完整结果保存到 `config/experiments/lc10a_weight_intervention.json` 和 `config/experiments/looming_weight_intervention.json`；实验说明位于 `experiments/`。

Decoder 鲁棒性实验覆盖 100/180/260/400/600 ms 窗口、假 spike、基线漂移、随机输出神经元损失、留出种子泛化、规则 Decoder 与 1,314 个 descending-neuron 线性 Readout，并对每种窗口运行真实闭环避碰：

```powershell
conda run -n flybrain python scripts\run_decoder_robustness.py --device cuda --batch 8 --train-seeds 4101,4201,4301 --test-seeds 5101,5201
```

机器结果为 `config/experiments/decoder_robustness.json`，说明见 `experiments/DECODER_ROBUSTNESS.md`。

## 温度、湿度和光照代理实验

开放刺激实验已验证 `TRN_VP2/VP3`、`HRN_VP4/VP5` 和 `OCG` 光照代理可写入当前 FlyBrain，并比较强度、变化率、spike probability 和自适应阈值编码。全体 1,314 个 descending neurons 的后续扫描没有为所有通道找到兼具稳定性与明确行为依据的输出，因此这些新感觉通道尚未接入 Decoder 或运动。

```powershell
conda run -n flybrain python scripts\run_multisensory_encoding.py --device cuda --trials-per-level 8 --seeds 7101,7203,7307
conda run -n flybrain python scripts\scan_sensory_descending.py --device cuda --batch 8 --seeds 8101,8203,8307
```

结果与边界见 `experiments/MULTISENSORY_ENCODING_AND_DN_SCAN.md`。

DNb05 的后续因果消融使用留出种子和排除 DNb05 自身的下降神经元读出。实验质量门通过，但 0× 并未显著破坏温湿度表征，因此不支持把 DNb05 视为当前模型中的必要瓶颈，也没有把它接入运动：

```powershell
conda run -n flybrain python scripts\run_dnb05_ablation.py --device cuda --batch 8 --train-seeds 9101,9203,9307 --test-seeds 10101,10203
```

## 局部奖励可塑性实验

LC10a 出站突触的受限奖励调制时序实验包含 Frozen、Sham、随机奖励和相关奖励四个条件。权重确实改变，但效果只在左侧明显，且没有以预设余量优于随机奖励或 Frozen，因此实验质量门失败；当前不能声称模型已经学会转向，也未进入联合训练或实时闭环学习。

```powershell
conda run -n flybrain python scripts\run_reward_plasticity.py --device cuda --batch 8 --epochs 4 --train-seed 11101 --test-seeds 12101,12203,12307
```

结果和解释边界见 `experiments/DNB05_ABLATION_AND_REWARD_PLASTICITY.md`。

局部无监督 STDP 另设 Frozen、跨个体打乱配对和同个体因果时序三条件。因果与打乱权重倍率的 RMS 差为 0.0974，留出方向准确率没有下降，全脑活动未饱和，实验质量门通过。该结果仅证明当前计算规则具有时序特异性，不证明真实 LC10a 突触采用这一机制：

```powershell
conda run -n flybrain python scripts\run_local_stdp.py --device cuda --batch 8 --epochs 4 --train-seed 13101 --test-seeds 14101,14203,14307
```

结果与解释边界见 `experiments/LOCAL_STDP.md`。

固定/适应 FlyBrain 与固定/重训线性 Readout 的联合比较也已完成。STDP 后重训 Readout 的留出准确率为 95.8%，但 STDP 后使用冻结 Readout 只有 79.2%，低于预设 80% 稳定门，因此总体实验未通过；不能把结果描述成已经获得稳健共同适应：

```powershell
conda run -n flybrain python scripts\run_joint_adaptation.py --device cuda --batch 8 --epochs 4 --train-seed 15101 --readout-train-seeds 16101,16203,16307 --test-seeds 17101,17203,17307
```

结果见 `experiments/JOINT_ADAPTATION.md`。

物理温湿度/光照传感器尚未接入。当前 Windows 设备审计没有发现串口或匹配的传感器/微控制器，因此项目不会用软件场数据冒充硬件读数。连接设备后可先重新生成清单：

```powershell
pwsh -NoProfile -File scripts\audit_sensor_hardware.ps1
```

当前证据保存在 `config/experiments/sensor_hardware_inventory.json`；实际适配仍需要设备型号、串口/USB 协议、采样率和校准方法。

## 环境要求

- Windows 11 / PowerShell 7
- Conda 环境 `flybrain`，Python 3.11.16
- GPU 路线：NVIDIA GPU/驱动、CuPy 14.2.0 和 CUDA 12.9 运行组件
- Node.js/npm（本机验证版本为 26.10.0 / 12.1.0，只用于安装 Three.js 0.186.0 静态依赖）
- FlyBrain 连接组数据 `brain.npz` 与 `weights.npz`（约 260 MB，首次使用自动下载）

完整的系统要求、基础/GPU 包清单、CPU 安装路线、数据校验和排错见 [ENVIRONMENT.md](ENVIRONMENT.md)。

依赖已经在当前工作区安装时，可直接启动：

```powershell
Set-Location -LiteralPath 'E:\neuro\fly_demo'
.\start.ps1
```

浏览器默认打开 <http://127.0.0.1:8765>。

## 首次安装或更新依赖

首次安装，在项目根目录创建 GPU 环境：

```powershell
Set-Location -LiteralPath 'E:\neuro\fly_demo'
conda env create -f environment.yml
npm ci
conda run -n flybrain python -c "from flybrain.data import ensure_data; print(ensure_data())"
conda run -n flybrain python -m pip check
.\start.ps1
```

已有 `flybrain` 环境时，用 `conda run -n flybrain python -m pip install -r requirements-cuda.txt`
同步 GPU 依赖，再运行 `npm ci`。`requirements.txt` 仅包含基础/CPU 依赖；
现有 `start.ps1` 和 `calibrate.ps1` 明确使用 CUDA。

## 重新校准

```powershell
.\calibrate.ps1
```

校准使用 8 个并行模拟个体，分别测量 baseline、左右 `LC10a` 和左右 looming
响应，并写入 `config/decoder.json`。

## 操作

- 点击地面：放置仅用于视觉诊断的目标；不再是自主运行的前提。
- “移动目标”：让目标围绕原点移动。
- “左/中央/右侧威胁”：生成向果蝇逼近的捕食者阴影，产生 looming 输入。
- 相机支持自由、跟随、果蝇第一视角和顶视；自由视角可拖动和滚轮缩放。
- 右侧四层诊断与十秒时间线均可折叠，顶栏也可整体收起诊断区。
- 直接刺激按钮：绕过世界 Encoder，便于验证特定 population。
- 暂停后可执行单步 100 ms。
- 当前会话以 JSONL 写入 `logs/`，面板可直接下载。
- 单次日志上限为 20 MB，达到上限后停止追加，避免长期运行占满磁盘。

## 测试

```powershell
conda run -n flybrain python -m unittest discover -s tests -v
```

服务器运行时可执行端到端探针；它会验证 DNa02 转向、ORN 偏转、DNp01 逃逸、三维起飞和一次闭环避碰，再重置场景：

```powershell
conda run -n flybrain python scripts\probe_server.py
```
