# FlyBrain–Decoder 联合适应实验

更新日期：2026-09-22

## 设计

本实验在相同数据划分下比较五个分支：

1. 冻结 FlyBrain + 规则 Decoder；
2. STDP 后 FlyBrain + 规则 Decoder；
3. 冻结 FlyBrain + 训练线性 Readout；
4. STDP 后 FlyBrain + 冻结的原始线性 Readout；
5. STDP 后 FlyBrain + 重新训练线性 Readout。

STDP 仍只修改 275 个 LC10a 的 39,282 条一级出站突触，规则和边界与 `LOCAL_STDP.md` 相同。线性 Readout 使用 1,314 个下降神经元 trace、PCA 和 L2 logistic regression；训练种子 16101/16203/16307，测试种子 17101/17203/17307，完全不重叠，每个种子 8 个并行个体。

这里的“联合适应”指先做局部无监督 STDP，再在适应后的网络 trace 上重新拟合 Readout。它不是同时在线共同学习，也不包含已验证的生物奖励回路。

## 结果

| 分支 | 留出准确率 | AUC 或转向幅度 |
|---|---:|---:|
| 冻结脑 + 规则 Decoder | 85.4% | 转向幅度 0.304 |
| STDP 脑 + 规则 Decoder | 95.8% | 转向幅度 0.404 |
| 冻结脑 + 训练 Readout | 85.4% | AUC 0.917 |
| STDP 脑 + 冻结 Readout | 79.2% | AUC 0.925 |
| STDP 脑 + 重训 Readout | 95.8% | AUC 0.983 |

联合分支比最佳单适应 Readout 分支高 10.4 个百分点，达到预设的 2 个百分点优势标准。但 STDP 后网络交给冻结 Readout 时只有 79.2%，低于预设 80% 稳定门，因此总体 `experiment_passed = false`。

这组结果支持一个有限解释：局部权重变化使下降神经元特征分布发生漂移，重新训练 Readout 可以补偿这一漂移；它尚不能证明当前联合流程在未重训 Decoder 时稳定，也不能证明生物学上的共同适应。

STDP 后全脑平均活动从 2.336 变为 2.345 Hz/神经元，单步峰值放电比例为 0.0612，没有全局饱和。原始 `weights.npz` 实验前后 SHA-256 一致。

## 与奖励对照的关系

Frozen、Sham reward、Random reward 和 Contingent reward 已在 `reward_plasticity.json` 中单独运行。相关奖励实验未通过双侧学习与随机奖励分离门，因此本实验不把联合 Readout 的改善归因于奖励学习。

## 复现

```powershell
conda run -n flybrain python scripts\run_joint_adaptation.py --device cuda --batch 8 --epochs 4 --train-seed 15101 --readout-train-seeds 16101,16203,16307 --test-seeds 17101,17203,17307
```

机器结果：`config/experiments/joint_adaptation.json`。
