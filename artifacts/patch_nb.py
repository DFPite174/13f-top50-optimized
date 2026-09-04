import json
from pathlib import Path

nb_path = Path("notebooks/13F_Top50_optimized.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update cell 1 (the first code cell)
nb["cells"][1]["source"] = [
    "# 单元 1：环境自适应配置与包路径导入\n",
    "import os, sys\n",
    "from pathlib import Path\n",
    "\n",
    "# 若在 Google Colab 云端直接打开本 Notebook，自动拉取仓库代码并安装依赖\n",
    "if 'google.colab' in sys.modules:\n",
    "    if not os.path.exists('13f-top50-optimized'):\n",
    "        !git clone https://github.com/DFPite174/13f-top50-optimized.git\n",
    "    %cd 13f-top50-optimized\n",
    "    !pip install -q edgartools yfinance\n",
    "\n",
    "PROJECT_ROOT = Path('.').resolve().parent if Path('.').resolve().name == 'notebooks' else Path('.').resolve()\n",
    "SRC_DIR = PROJECT_ROOT / 'src'\n",
    "if str(SRC_DIR) not in sys.path:\n",
    "    sys.path.insert(0, str(SRC_DIR))\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from top50_strategy.config import RunConfig\n",
    "from top50_strategy.pipeline import run_research, SyntheticAdapters\n",
    "\n",
    "print('✓ top50_strategy 模块与研究环境就绪。')"
]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)
print("Updated notebook with Colab self-adapter.")