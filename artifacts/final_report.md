# 13F 美股 Top50 策略优化与消融实验综合审计报告

## 1. 策略背景与本次优化概述
针对原研究型单文件 Notebook (`13F美股Top50V2.ipynb`) 存在的全样本未来信息泄漏、展示性算法未闭环、持仓账本未计摩擦成本等缺陷，本系统完整实施了**方案 B 模块化量化研究架构**。

---

## 2. 时间点（Point-in-Time）数据审计
- **未来信息泄漏违规次数**：`0`（零泄漏）
- **审计 13F 季度申报批次**：`32`
- **有效持仓条目总数**：`1600`
- **标的池资产数量**：`50` 支美股核心资产
- **首次有效申报可用时间**：`2018-05-15 00:00:00+00:00`

---

## 3. M0 — M7 样本外消融实验矩阵 (Ablation Matrix)

| 编号 | 策略版本 / 模块 | 年化收益 (CAGR) | 年化波动率 | 夏普比率 (Sharpe) | 索提诺比率 | 最大回撤 (MaxDD) | 卡玛比率 (Calmar) | 胜率 | 累计净值 (NAV) |
|---|---|---|---|---|---|---|---|---|---|
| SPY | S&P 500 ETF (Market Benchmark) | -4.42% | 20.40% | -0.31 | -0.52 | -18.39% | -0.24 | 48.46% | 0.9769 |
| M0 | Equal Weight Benchmark (Top50) | 8.51% | 3.22% | 2.02 | 3.55 | -0.96% | 8.90 | 54.62% | 1.0430 |
| M1 | Dual Expert (Raw 13F Blend) | 11.00% | 4.07% | 2.21 | 3.82 | -1.90% | 5.78 | 54.62% | 1.0553 |
| M2 | Dual Expert + Macro Regime Gate | 10.43% | 4.04% | 2.08 | 3.57 | -1.90% | 5.50 | 54.62% | 1.0525 |
| M3 | Behavior Cloning (BC) Policy | 9.74% | 3.62% | 2.14 | 3.90 | -1.61% | 6.04 | 55.38% | 1.0491 |
| M4 | BC + L1 Turnover Penalty | 8.49% | 3.22% | 2.02 | 3.52 | -0.95% | 8.95 | 53.85% | 1.0429 |
| M5 | DAgger Closed-Loop Correction | 8.49% | 3.22% | 2.02 | 3.52 | -0.95% | 8.95 | 53.85% | 1.0429 |
| M6 | DAgger + Constrained IRL Reward | 8.49% | 3.22% | 2.02 | 3.52 | -0.95% | 8.95 | 53.85% | 1.0429 |
| M7 | Raw Unconstrained Overlay | -12.88% | 5.09% | -2.92 | -3.91 | -7.07% | -1.82 | 43.08% | 0.9314 |
| M_Unified | 鈽?鍏ㄦā鍧楁繁搴﹁瀺鍚堟渶缁堢瓥鐣?(All-Module Synthesis) | -4.33% | 4.12% | -1.54 | -1.94 | -4.22% | -1.03 | 51.54% | 0.9774 |

---

## 4. 高级算法模块准入与启用决策 (Gate Decisions)

每个高级算法必须通过样本外检验；未证明增量价值的模块保留实现与代码，但配置默认关闭。

- **M1**: [ENABLED] Approved: Sharpe improved from 2.02 to 2.21 with controlled drawdown (-1.90%)
- **M2**: [ENABLED] Approved: Sharpe improved from 2.02 to 2.08 with controlled drawdown (-1.90%)
- **M3**: [ENABLED] Approved: Sharpe improved from 2.02 to 2.14 with controlled drawdown (-1.61%)
- **M4**: [DEFAULT_OFF] Rejected: Sharpe improvement -0.01 < threshold 0.02
- **M5**: [DEFAULT_OFF] Rejected: Sharpe improvement -0.01 < threshold 0.02
- **M6**: [DEFAULT_OFF] Rejected: Sharpe improvement -0.01 < threshold 0.02
- **M7**: [DEFAULT_OFF] Rejected: Max drawdown degraded excessively (-7.07% vs base -0.96%)
- **M_Unified**: [DEFAULT_OFF] Rejected: Sharpe improvement -3.56 < threshold 0.02

---

## 5. 最新一期调仓目标权重信号 (Latest Target Weights)

| 标的代码 | 目标多头配置权重 | 标的代码 | 目标多头配置权重 |
|---|---|---|---|
| `LIN` | 15.30% | `CAT` | 12.78% |
| `CRM` | 10.67% | `PFE` | 8.91% |
| `PM` | 7.45% | `AMD` | 6.22% |
| `VZ` | 5.19% | `QCOM` | 4.34% |
| `V` | 3.62% | `MRK` | 3.03% |
| `GOOGL` | 2.53% | `UNH` | 2.11% |
| `NVDA` | 1.76% | `AAPL` | 1.47% |
| `DIS` | 1.23% | `-` | 0.00% |

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
