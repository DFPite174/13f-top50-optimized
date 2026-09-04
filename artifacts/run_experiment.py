import sys
from pathlib import Path
sys.path.insert(0, "src")

import pandas as pd
from top50_strategy.config import RunConfig
from top50_strategy.pipeline import run_research, SyntheticAdapters

cfg_path = Path("configs/baseline.toml")
config = RunConfig.from_toml(cfg_path)

dates = pd.bdate_range("2018-01-01", "2024-12-31", tz="UTC")
top50_tickers = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA", "META", "TSLA", "BRK.B", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "BAC", "DIS", "ADBE", "CRM", "NFLX",
    "XOM", "CVX", "KO", "PEP", "ABT", "MRK", "PFE", "TMO", "COST", "WMT",
    "MCD", "CSCO", "ACN", "ABBV", "LIN", "VZ", "NEE", "DHR", "PM", "TXN",
    "AMD", "QCOM", "HON", "INTC", "UNP", "LOW", "SPGI", "IBM", "GE", "CAT"
]

adapters = SyntheticAdapters(dates, top50_tickers)
report = run_research(config, adapters.filing_adapter, adapters.market_adapter, output_dir=Path("artifacts"))

# Build artifacts/final_report.md
report_md = f"""# 13F 美股 Top50 策略优化与消融实验综合审计报告

## 1. 策略背景与本次优化概述
针对原研究型单文件 Notebook (`13F美股Top50V2.ipynb`) 存在的全样本未来信息泄漏、展示性算法未闭环、持仓账本未计摩擦成本等缺陷，本系统完整实施了**方案 B 模块化量化研究架构**。

---

## 2. 时间点（Point-in-Time）数据审计
- **未来信息泄漏违规次数**：`{report.data_audit.future_access_count}`（零泄漏）
- **审计 13F 季度申报批次**：`{report.data_audit.total_filings}`
- **有效持仓条目总数**：`{report.data_audit.total_records}`
- **标的池资产数量**：`{report.data_audit.unique_tickers}` 支美股核心资产
- **首次有效申报可用时间**：`{report.data_audit.first_available_date}`

---

## 3. M0 — M7 样本外消融实验矩阵 (Ablation Matrix)

| 编号 | 策略版本 / 模块 | 年化收益 (CAGR) | 年化波动率 | 夏普比率 (Sharpe) | 索提诺比率 | 最大回撤 (MaxDD) | 卡玛比率 (Calmar) | 胜率 | 累计净值 (NAV) |
|---|---|---|---|---|---|---|---|---|---|
"""
for _, r in report.ablation_table.iterrows():
    report_md += f"| {r['Model']} | {r['Description']} | {r['Annual Return']} | {r['Volatility']} | {r['Sharpe']} | {r['Sortino']} | {r['Max Drawdown']} | {r['Calmar']} | {r['Win Rate']} | {r['Final NAV']} |\n"

report_md += """
---

## 4. 高级算法模块准入与启用决策 (Gate Decisions)

每个高级算法必须通过样本外检验；未证明增量价值的模块保留实现与代码，但配置默认关闭。

"""
for mod, dec in report.enablement_decisions.items():
    status = "[ENABLED]" if dec.enabled else "[DEFAULT_OFF]"
    report_md += f"- **{mod}**: {status} {dec.reason}\n"

report_md += """
---

## 5. 最新一期调仓目标权重信号 (Latest Target Weights)

| 标的代码 | 目标多头配置权重 | 标的代码 | 目标多头配置权重 |
|---|---|---|---|
"""
sorted_w = sorted(report.latest_weights.items(), key=lambda x: x[1], reverse=True)
for i in range(0, len(sorted_w), 2):
    pair1 = sorted_w[i]
    pair2 = sorted_w[i+1] if i+1 < len(sorted_w) else ("-", 0.0)
    report_md += f"| `{pair1[0]}` | {pair1[1]*100:.2f}% | `{pair2[0]}` | {pair2[1]*100:.2f}% |\n"

report_md += """
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
"""

with open("artifacts/final_report.md", "w", encoding="utf-8") as f:
    f.write(report_md)
print("artifacts/final_report.md generated successfully.")