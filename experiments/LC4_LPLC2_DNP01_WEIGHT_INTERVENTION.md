# LC4/LPLC2 → DNp01 出站权重干预实验

## 结论

在当前 FlyBrain LIF 模型中，双侧 LC4/LPLC2 全部出站连接的权重倍率对同侧 DNp01 放电率和 Decoder action 具有明确因果影响。DNp01 放电率随 `0× → 0.5× → 1× → 2×` 单调增加；当前 Decoder 阈值把 0.5× 的亚阈值神经响应与 1× 的稳定逃逸授权区分开来。

## 实验设计

- 条件：`Baseline / Sham 1× / 0× / 0.5× / 2×`。
- 干预：双侧 LC4/LPLC2 的全部出站突触，只在运行进程内缩放。
- 试验：左右 looming 各 3 个种子 × 8 个并行个体，即每侧每条件 24 次。
- 时序：1 秒 warmup、1 秒 0.8 强度刺激；50 Hz 神经模拟、260 ms 窗口、10 Hz Decoder。
- 行为输出：DNp01 action 是否触发、首次触发时间和 action 占控制周期比例。
- 无 action 的延迟以 1.1 秒右删失值记录，不伪装成实际反应时间。

## 主要结果

| 条件 | 同侧 DNp01（Hz） | action 触发率 | 首次 action 或删失时间（s） |
|---|---:|---:|---:|
| 0× | 0.063 | 0% | 1.1 |
| 0.5× | 20.854 | 0% | 1.1 |
| Baseline | 46.167 | 100% | 0.1 |
| Sham 1× | 46.208 | 100% | 0.1 |
| 2× | 49.188 | 100% | 0.1 |

- DNp01 剂量关系、action 触发率和 action 延迟均通过单调性检查。
- Baseline 与 Sham 的 DNp01 均值仅差 0.042 Hz，action 触发率完全相同。
- `0×` 相对 Baseline 降低 46.104 Hz。
- `2×` 全脑单步峰值放电比例为 0.062933，低于 0.10 饱和界限。
- 原始 `weights.npz` 的实验前后 SHA-256 一致。
- 总体质量门 `experiment_passed = true`。

0.5× 结果说明 Decoder 阈值具有实际作用：神经活动并非为零，但约 20.9 Hz 的 DNp01 均值不足以越过当前归一化 action 阈值。1× 已接近单神经元 50 Hz 的步长上限，因此 2× 主要表现为小幅逼近上限，而不是成倍增长。

## 可复现资产

- 实验脚本：`scripts/run_escape_weight_intervention.py`
- 公共内存干预器：`app/interventions.py`
- 完整结果：`config/experiments/looming_weight_intervention.json`

```powershell
Set-Location -LiteralPath 'E:\neuro\fly_demo'
conda run -n flybrain python scripts\run_escape_weight_intervention.py --device cuda --batch 8 --seeds 1101,2203,3307
```

## 因果边界

1. 干预覆盖 LC4 和 LPLC2 的全部出站连接，不代表一条直接单突触通路。
2. DNp01 action 只授权急停/逃逸；实际垂直方向仍由明确标注的非神经世界策略选择。
3. 该开放刺激实验验证神经响应和 action，不等同于完整世界中的碰撞避免成功率。
4. 结果支持当前模拟模型内部的因果关系，不直接给出真实果蝇效应大小。

## 下一步

Decoder 鲁棒性实验应优先检查 0.5× 与 1× 之间的阈值稳定性，包括时间窗口、背景漂移、spike 丢失、随机神经元损失和跨随机种子误触发率。
