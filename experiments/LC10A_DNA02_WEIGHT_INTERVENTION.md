# LC10a → DNa02 出站权重干预实验

## 结论

在当前 FlyBrain LIF 模型中，双侧 LC10a 全部出站突触的权重倍率与同侧 DNa02 放电率、Decoder 转向幅度均呈单调剂量关系。`0×` 明显削弱响应，`2×` 增强响应，同时没有观察到全脑饱和。该结果支持 LC10a 出站连接对当前模型转向链路具有因果贡献。

本结论只适用于当前 connectome 权重构造和 LIF 动力学，不能直接解释为真实果蝇中的生物学效应大小。

## 实验设计

- 条件：`Baseline / Sham 1× / 0× / 0.5× / 2×`。
- 干预：双侧 275 个 LC10a 的全部 39,282 条出站连接。
- 试验：每个条件分别刺激左、右 LC10a；每侧 3 个种子 × 8 个并行个体，共 24 次。
- 时序：1 秒 warmup，随后以 0.8 强度刺激 1 秒；神经模拟 50 Hz，Decoder 窗口 260 ms、控制输出 10 Hz。
- 输出：DNa02 左右放电率、Decoder 转向、DNp01 意外动作、全脑 spike、单步放电比例和运行倍率。
- 隔离：每个倍率均从同一份内存基线权重重新生成，不累乘；原始 `weights.npz` 只读并在实验前后核对 SHA-256。

`Sham` 与 Baseline 使用相同权重和随机种子。CUDA 稀疏传播的两次运行没有逐位相同，因此使用预先写入结果文件的等效界限：同侧 DNa02 均值差不超过 0.25 Hz、转向幅度差不超过 0.02、方向正确率差不超过 0.05。

## 主要结果

| 条件 | 同侧 DNa02 均值（Hz） | Decoder 转向幅度 | 说明 |
|---|---:|---:|---|
| 0× | 0.417 | 0.071 | 只剩基线噪声和网络背景；方向不再可靠 |
| 0.5× | 1.750 | 0.248 | 响应减弱 |
| Baseline | 4.021 | 0.640 | 左右方向正确率 100% |
| Sham 1× | 4.042 | 0.641 | 与 Baseline 在等效界限内 |
| 2× | 7.729 | 0.947 | 响应增强 |

- `0×` 与 Baseline 的同侧 DNa02 差值为 3.604 Hz。
- DNa02 放电率和转向幅度均通过单调性检查。
- `2×` 的全脑单步峰值放电比例为 0.05961，低于预设的 0.10 饱和界限。
- Baseline 单实例实时倍率为 0.926；五个条件的批处理运行倍率约为 0.93–1.04。
- 原始权重实验前后 SHA-256 均为 `c29919aa44069a271b1ee978abe05fa9bf6e45e4ba3e436e92b624ef1b5be40c`。
- 总体质量门 `experiment_passed = true`。

左右分项中，Baseline 的左目标平均最终转向为 -0.694，右目标为 +0.586；`2×` 分别增强至 -0.963 和 +0.932。`0×` 下仍存在少量 DNa02 自发活动，因此它代表切断 LC10a 出站贡献，而不是将整个转向网络静音。

## 可复现资产

- 实验脚本：`scripts/run_weight_intervention.py`
- 内存干预实现：`app/interventions.py`
- 单元测试：`tests/test_interventions.py`
- 完整机器可读结果：`config/experiments/lc10a_weight_intervention.json`

运行命令：

```powershell
Set-Location -LiteralPath 'E:\neuro\fly_demo'
conda run -n flybrain python scripts\run_weight_intervention.py --device cuda --batch 8 --seeds 1101,2203,3307
```

## 因果边界

1. 干预的是 LC10a 的全部出站连接，不是假设存在一条直接的 LC10a→DNa02 单突触连接。
2. 转向指标是当前 Decoder 命令的时间积分，不是肌肉或真实飞行力矩。
3. 本实验没有修改 Decoder 参数，因此剂量效应来自 FlyBrain 内部传播变化，而不是重新拟合 Decoder。
4. 本实验没有改变原始数据文件；退出进程后干预自动消失。

## 下一步

用同一框架对 `LC4/LPLC2 → DNp01 → escape` 做五条件实验。该实验应增加 action 首次触发延迟、错误触发率和闭环避碰率，随后再进入 Decoder 窗口与噪声鲁棒性实验。
