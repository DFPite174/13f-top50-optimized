import json
from pathlib import Path

nb_path = Path("notebooks/13F_Top50_optimized.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Cell 5 update: print ablation table with SPY and M_Unified
nb["cells"][4]["source"] = [
    "# 单元 5：展示 M0 — M7 及全模块融合策略与基准 SPY 对比表\n",
    "print('============================ 策略与标普500(SPY)基准样本外消融对比表 ============================')\n",
    "display_df = report.ablation_table.copy()\n",
    "print(display_df.to_string(index=False))\n",
    "print('==============================================================================================')"
]

# Cell 7 update: show tiered conviction weights
nb["cells"][6]["source"] = [
    "# 单元 7：导出全模块深度融合策略最新一期梯度持仓信号 (Conviction Tiered Allocation)\n",
    "print('======================= 最新实盘/离线调仓目标信号 (Conviction Tiered Allocation) =======================')\n",
    "sorted_weights = sorted(report.latest_weights.items(), key=lambda x: x[1], reverse=True)\n",
    "for rank, (ticker, w) in enumerate(sorted_weights, 1):\n",
    "    tier = '【核心底仓】' if w >= 0.10 else ('【主力加仓】' if w >= 0.05 else '【卫星配置】')\n",
    "    print(f'  > 排名 {rank:>2} | 标的 {ticker:<6}: 权重 {w*100:6.2f}%  {tier}')\n",
    "print('=====================================================================================================')"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print("Updated notebook cells for SPY and conviction tiering.")