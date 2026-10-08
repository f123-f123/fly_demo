# 温度、湿度和光照代理编码与下降神经元扫描

## 结论

温度、湿度和光照代理都能稳定写入当前 FlyBrain，并在对应上行 population 中形成可重复的强度响应。强度编码和 spike-probability 编码表现最好；单步 rate-of-change 编码只产生短暂响应，1 秒平均 SNR 较低。

全 descending-neuron 扫描没有为全部通道找到同时满足稳定性和行为依据的输出：干燥通道没有候选通过神经稳定性门；其他通道候选存在交叉响应或缺少与趋温、趋湿、光趋性相匹配的已验证行为角色。因此本阶段不把新感觉通道接入 Decoder 或运动。

## 感觉入口

| 软件通道 | 注入 population | 上行读出 |
|---|---|---|
| 升温 | `TRN_VP2` | VP2 PN 组 |
| 降温 | `TRN_VP3a/b` | VP3 PN 组 |
| 干燥 | `HRN_VP4` | VP4 PN 组 |
| 潮湿 | `HRN_VP5` | VP5 PN 组 |
| 光照代理 | `OCG01/02` | `DNp20/DNp22/OCC02b` 强下游组 |

`TRN` 和 `HRN` 在当前数据中属于 `cb_sensory`。未找到可确认的原始 ocellar photoreceptor 类型，因此光照不是从“原始感光细胞”开始，而是写入 `OCG` ocellar projection population；它只能称为光照接口代理。

## 编码实验

- 编码器：强度、单步变化率、spike probability、自适应阈值。
- 归一化强度：`0 / 0.25 / 0.5 / 0.75 / 1.0`。
- 每个曲线点：3 个种子 × 8 个个体，共 24 个样本。
- 每次刺激 1 秒；不读取或控制运动。

### 最大强度的目标输出结果

| 通道 | 编码器 | 输出信号（Hz） | SNR | 通道分离比 | 跨种子 CV | 平均首次响应 |
|---|---|---:|---:|---:|---:|---:|
| 升温 | spike probability | 12.54 增量 | 79.85 | 6.47 | 0.012 | 27.5 ms |
| 降温 | spike probability | 19.03 增量 | 54.51 | 1.78 | 0.040 | 20.0 ms |
| 干燥 | spike probability | 15.74 增量 | 41.67 | 125.94 | 0.019 | 22.5 ms |
| 潮湿 | spike probability | 13.83 增量 | 51.40 | 3.72 | 0.013 | 23.3 ms |
| 光代理 | spike probability | 4.28 增量 | 20.59 | 42.79 | 0.020 | 34.2 ms |

强度编码也为所有通道产生单调曲线。自适应阈值输出较弱但稳定。单步变化率的 SNR 为升温 2.41、降温 1.24、干燥 0.68、潮湿 1.93、光代理 1.04，说明一次 20 ms 脉冲不适合作为当前 1 秒平均读出的主编码方式；这不排除它对瞬时检测的价值。

软件信号尚未标定为摄氏度、相对湿度或 lux，因而当前结果是规范化刺激—响应曲线，不是物理传感器标定。

## 全下降神经元扫描

- 范围：1,314 个 descending neurons。
- 条件：每通道 3 个种子 × 8 个个体，恒定强度 0.8，刺激 1 秒。
- 稳定候选门：相对 baseline `Δ ≥ 1 Hz`、效应量 `≥ 2`、每个种子的平均变化均为正。

| 通道 | 稳定候选数 | 领先候选 | 主要问题 |
|---|---:|---|---|
| 升温 | 2 | `DNc02`, `DNp12` | DNp12 跨通道；DNc02 功能宽泛，非温度专用 |
| 降温 | 5 | `DNb05`, `DNp12`, `DNge141` | DNb05 与潮湿交叉响应 |
| 干燥 | 0 | 无 | 未通过神经稳定性门 |
| 潮湿 | 6 | `DNb05`, `DNp12`, `DNp25`, `DNg56` | 与降温共享多个候选 |
| 光代理 | 23 | `DNp19`, `DNp16`, `DNg79`, `DNp05` | 缺少与本光代理输入相匹配的确定行为映射 |

主要模拟响应：升温 `DNc02_R Δ=1.67 Hz`；降温 `DNb05_L/R Δ=3.63/3.46 Hz`；潮湿 `DNb05_R Δ=3.46 Hz`；光代理 `DNp19_R/L Δ=7.92/5.63 Hz`。干燥最强候选 `DNc02_R` 只有 `Δ=0.875 Hz、效应量=1.23`，未达到门限。

## 生物学角色审计

- DNb05 的双侧差与行走转向速度相关，但它并非已证明的趋温或趋湿专用命令神经元：[Fine-grained descending control of steering in walking Drosophila](https://pmc.ncbi.nlm.nih.gov/articles/PMC10614758/)。
- 热威胁研究把 DNb05 列为潜在热感觉下行链路和快速运动相关 DN，但该证据指向威胁/运动，不足以定义温度偏好控制：[Rapid threat assessment in the Drosophila thermosensory system](https://www.nature.com/articles/s41467-023-42864-5)。
- DNc02 的形态覆盖广泛 VNC neuropils，已有资料没有把它定义为温度专用输出：[The functional organization of descending sensory-motor pathways in Drosophila](https://pmc.ncbi.nlm.nih.gov/articles/PMC6019073/)。
- 当前综述性 connectome 比较明确列出 DNp15/20/22 与飞行和颈部控制，但没有为本扫描领先的 DNp19/DNp16/DNg79 提供同等明确、与光照匹配的角色：[Comparative connectomics of Drosophila descending and ascending neurons](https://www.nature.com/articles/s41586-025-08925-z)。

结论是：DNb05 值得作为后续温度/湿度因果消融候选，但不能直接用作趋温/趋湿 Decoder；光照候选也只能继续审计，不能直接接运动。

## 可复现资产

- 编码比较：`scripts/run_multisensory_encoding.py`
- 编码结果：`config/experiments/multisensory_encoding.json`
- DN 扫描：`scripts/scan_sensory_descending.py`
- DN 扫描结果：`config/experiments/sensory_descending_scan.json`

```powershell
conda run -n flybrain python scripts\run_multisensory_encoding.py --device cuda --trials-per-level 8 --seeds 7101,7203,7307
conda run -n flybrain python scripts\scan_sensory_descending.py --device cuda --batch 8 --seeds 8101,8203,8307
```

## 阶段决策

按预先约定的科学边界，本阶段停在神经输入实验：不将 PN、环境场或未经验证的 DN 直接连接到运动。下一项合理实验是对 DNb05 做温度/湿度刺激下的 `Baseline / Sham / 0×` 因果消融，验证它是否只是相关响应，还是当前模型中的必要瓶颈。
