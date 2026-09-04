# 13F 美股 Top50 策略模块化研究系统 (方案 B 独立工程)

本项目是按照《13F 美股 Top50 V2 策略优化设计规格》完全独立研发的量化策略研究与回测系统。

---

## 核心设计与研究原则

1. **严格时间点（Point-in-Time）数据**：
   - 彻底清除未来信息泄漏（Future-leak free）。
   - 股票池按每次模型重训日前已披露的最近 8 个季度 13F 申报动态筛选 Top 50，禁止全时段统计未来报告。
   - 行情特征严格对齐：T 日收盘特征只能在 T+1 日开盘/收盘生成与执行，严禁 `bfill()` 或用未来价格填充缺失值。
   - 标准化器（`StandardScaler`）仅在训练集上 `fit`，绝不接触验证集或测试集。

2. **真实持仓账本与执行模拟器**：
   - 显式追踪多头持仓、空头持仓、现金与债务，严格遵循会计恒等式：`Equity = Long + Short + Cash - Debt`。
   - 纳入全部摩擦成本：交易佣金（2.5 bps）、买卖滑点（4.0 bps）、融券做空借券费率（年化 2.0%）与融资利率（年化 4.0%）。
   - 真实隔夜结转与收益归因，不再强行使用每日等权归一化抹平实际杠杆。

3. **双专家与宏观状态切换**：
   - 融合费雪投资（Fisher Investments，成长/Alpha 侧）与桥水基金（Bridgewater Associates，全天候/防御/ETF 侧）。
   - 提供基准固定门控（`HardRegimeGate`）与可微平滑门控（`SmoothRegimeGate`），基于标普 500、十年期美债收益率与 VIX 波动率动态分配两家机构的持仓权重。

4. **双头行为克隆策略网络（BC Policy）**：
   - 输入 103 维市场宏观特征 + 50 维当前持仓 + 现金比率 + 当前敞口（共 155 维）。
   - 双输出头：50 维资产配置概率（Softmax）与 1 维有界总风险敞口（Sigmoid 映射至 `[0.5x, 1.25x]`）。
   - 损失函数显式惩罚对真实当前持仓的 L1 换手率，并保持连续时序训练，不随机打乱样本。

5. **独立风控对冲层（Overlay）**：
   - Top-K 多头：根据模型得分截取前 15 大核心多头并应用牛市/正常/恐慌状态敞口缩放。
   - 条件式 Bottom-M 做空：**彻底与多头 Softmax 尾部解耦**。只有在宏观恐慌/破位、个股绝对动量为负、相对弱于大盘且不在保护性资产名单中时，才触发融券对冲。

6. **闭环交互式 DAgger 与最大熵奖励学习（IRL）**：
   - DAgger 的动作真实进入模拟器并改变后续访问状态，在策略真实偏离的持仓上查询专家动作，进行选择性样本聚合与去重。
   - 最大熵奖励学习（IRL）构建 `(state, action) -> reward` 模型；在验证集上检验判别能力，未达显著优势阈值时自动触发安全回退（Fallback to DAgger），默认关闭该模块。

---

## 目录结构

```text
13f-top50-optimized/
├── pyproject.toml                 # 依赖、pytest 与项目配置
├── README.md                      # 本说明文档
├── configs/
│   └── baseline.toml              # 冻结的默认研究参数
├── src/top50_strategy/
│   ├── __init__.py
│   ├── config.py                  # 配置数据类与有效性校验
│   ├── types.py                   # 跨模块数据契约与不变量
│   ├── data.py                    # 13F 申报解析、可用时点过滤与滚动股票池构建
│   ├── features.py                # 历史衍生特征与标准化（严禁向前填充）
│   ├── experts.py                 # 双专家融合与宏观门控
│   ├── simulator.py               # 真实持仓账本与摩擦成本结算模拟器
│   ├── policy.py                  # 双头行为克隆策略网络
│   ├── overlay.py                 # 独立风控对冲层（Top-K 多头与条件式 Bottom-M 对冲）
│   ├── dagger.py                  # 闭环交互式 DAgger 纠偏
│   ├── reward_learning.py         # 最大熵奖励学习与受约束微调
│   ├── evaluation.py              # 滚动前推、指标计算与 M0-M7 消融门控
│   └── pipeline.py                # 端到端研究流水线编排
├── tests/
│   ├── test_config_and_types.py   # 配置与不变量测试
│   ├── test_point_in_time_data.py # 点位数据与股票池测试
│   ├── test_experts_and_features.py # 特征与双专家测试
│   ├── test_simulator.py          # 模拟器账本与成本测试
│   ├── test_policy.py             # 策略网络与连续时序训练测试
│   ├── test_overlay.py            # 动作投影与独立做空测试
│   ├── test_dagger.py             # 真实 DAgger 状态转移测试
│   ├── test_reward_learning.py    # 奖励模型与回退保护测试
│   ├── test_evaluation.py         # 评估指标与消融测试
│   └── test_pipeline_smoke.py     # 离线端到端 Smoke 集成测试
├── notebooks/
│   └── 13F_Top50_optimized.ipynb  # 纯净交付薄 Notebook
└── artifacts/                     # 离线运行产物、质量报告与消融汇总表
```

---

## 运行与验证

### 1. 执行全量自动化测试（52 项测试）
```powershell
pytest tests -v
```

### 2. 打开薄 Notebook 交互体验
在 Jupyter Lab 或 VS Code 中直接打开 `notebooks/13F_Top50_optimized.ipynb`，点击“全部运行”即可查看点位数据审计、M0—M7 消融矩阵及最新调仓权重信号。