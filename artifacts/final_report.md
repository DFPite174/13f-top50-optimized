# 13F 美股 Top50 策略优化与消融实验综合审计报告

## 1. 策略背景与本次优化概述
针对原研究型单文件 Notebook (`13F美股Top50V2.ipynb`) 存在的全样本未来信息泄漏、展示性算法未闭环、持仓账本未计摩擦成本等缺陷，本系统完整实施了**方案 B 模块化量化研究架构**。

---

## 2. 时间点（Point-in-Time）数据审计
- **未来信息泄漏违规次数**：`0`（零泄漏）
- **审计 13F 季度申报批次**：`32`
- **有效持仓条目总数**：`1136`
- **标的池资产数量**：`50` 支美股核心资产
- **首次有效申报可用时间**：`2018-05-15 00:00:00+00:00`

---

## 3. M0 — M7 样本外消融实验矩阵 (Ablation Matrix)

| 编号 | 策略版本 / 模块 | 年化收益 (CAGR) | 年化波动率 | 夏普比率 (Sharpe) | 索提诺比率 | 最大回撤 (MaxDD) | 卡玛比率 (Calmar) | 胜率 | 累计净值 (NAV) |
|---|---|---|---|---|---|---|---|---|---|
| M0 | Equal Weight Benchmark | 8.51% | 3.22% | 2.02 | 3.55 | -0.95% | 8.93 | 54.62% | 1.0430 |
| M1 | Dual Expert (Raw 13F Blend) | 11.26% | 3.77% | 2.46 | 4.17 | -1.42% | 7.93 | 56.92% | 1.0566 |
| M2 | Dual Expert + Macro Regime Gate | 10.78% | 4.23% | 2.07 | 3.53 | -1.65% | 6.52 | 52.31% | 1.0542 |
| M3 | Behavior Cloning (BC) Policy | 10.02% | 3.58% | 2.24 | 3.80 | -1.36% | 7.36 | 56.92% | 1.0505 |
| M4 | BC + L1 Turnover Penalty | 8.46% | 3.23% | 2.00 | 3.51 | -0.96% | 8.83 | 54.62% | 1.0428 |
| M5 | DAgger Closed-Loop Correction | 8.46% | 3.23% | 2.00 | 3.51 | -0.96% | 8.83 | 54.62% | 1.0428 |
| M6 | DAgger + Constrained IRL Reward | 8.46% | 3.23% | 2.00 | 3.51 | -0.96% | 8.83 | 54.62% | 1.0428 |
| M7 | Full Dynamic Overlay (TopK + BottomM) | -8.31% | 199.15% | -0.05 | -0.07 | -34.53% | -0.24 | 44.62% | 0.9563 |

---

## 4. 高级算法模块准入与启用决策 (Gate Decisions)

每个高级算法必须通过样本外检验；未证明增量价值的模块保留实现与代码，但配置默认关闭。

- **M1**: [ENABLED] Approved: Sharpe improved from 2.02 to 2.46 with controlled drawdown (-1.42%)
- **M2**: [ENABLED] Approved: Sharpe improved from 2.02 to 2.07 with controlled drawdown (-1.65%)
- **M3**: [ENABLED] Approved: Sharpe improved from 2.02 to 2.24 with controlled drawdown (-1.36%)
- **M4**: [DEFAULT_OFF] Rejected: Sharpe improvement -0.02 < threshold 0.02
- **M5**: [DEFAULT_OFF] Rejected: Sharpe improvement -0.02 < threshold 0.02
- **M6**: [DEFAULT_OFF] Rejected: Sharpe improvement -0.02 < threshold 0.02
- **M7**: [DEFAULT_OFF] Rejected: Max drawdown degraded excessively (-34.53% vs base -0.95%)

---

## 5. 最新一期调仓目标权重信号 (Latest Target Weights)

| 标的代码 | 目标多头配置权重 | 标的代码 | 目标多头配置权重 |
|---|---|---|---|
| `PG` | 5.82% | `GOOGL` | 5.82% |
| `PFE` | 5.80% | `PEP` | 5.80% |
| `GE` | 5.80% | `BAC` | 5.79% |
| `CAT` | 5.79% | `TXN` | 5.79% |
| `MSFT` | 5.78% | `MA` | 5.78% |
| `NVDA` | 5.76% | `WMT` | 5.76% |
| `LOW` | 5.76% | `CSCO` | 5.76% |
| `NFLX` | 5.76% | `-` | 0.00% |

---

## 6. 与原 Notebook 的本质对比

| 维度 | 原 Notebook (`13F美股Top50V2.ipynb`) | 优化后模块化系统 (`top50_strategy`) |
|---|---|---|
| **股票池构建** | 使用全时段（2018-2024）统计，存在严重未来函数 | 严格基于重训日前 8 个季度已披露 13F 滚动构建 |
| **特征与标准化** | 依赖向后填充或固定值，全样本拟合归一化 | 仅历史滚动计算，缺失严格掩码，`fit` 严格受限于训练集 |
| **DAgger 纠偏** | 仅将历史数据重复拷贝放大，动作未影响持仓 | 闭环交互：策略动作改变模拟器真实状态，在偏离状态查询专家 |
| **IRL 奖励网络** | 仅做参数训练展示，未参与最终决策优化 | 最大熵对比学习，验证集不过关自动回退，微调受 KL 与换手约束 |
| **做空与对冲** | 粗暴取多头 Softmax 概率最低项 | 彻底解耦：仅在宏观恐慌 + 弱势破位 + 满足借券约束时启动 |
| **账本与交易成本** | 每日强行重新归一化，借券/融资成本缺失 | 显式追踪多头、空头、现金、负债，扣除 2.5bps 佣金、4bps 滑点与利息 |
| **工程形态** | 单文件脚本式 Notebook | 模块化 Python 包（10 个模块 + 10 组全量单元测试）+ 薄 Notebook |
