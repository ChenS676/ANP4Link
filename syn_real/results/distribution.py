import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Data
data = [
    ("BARABASI ALBERT", 2000, 1.000000, "Barabási-Albert"),
    ("BARABASI ALBERT", 1000, 1.000000, "Barabási-Albert"),
    ("BARABASI ALBERT", 500, 1.000000, "Barabási-Albert"),
    ("BARABASI ALBERT", 400, 0.800000, "Barabási-Albert"),
    ("BARABASI ALBERT", 150, 0.700000, "Barabási-Albert"),
    ("BARABASI ALBERT", 10, 0.400000, "Barabási-Albert"),
    ("GraphType.BARABASI_ALBERT", 100, 0.600000, "Barabási-Albert"),
    ("GraphType.BARABASI_ALBERT", 90, 0.500000, "Barabási-Albert"),
    ("GraphType.BARABASI_ALBERT", 80, 0.500000, "Barabási-Albert"),
    ("GraphType.BARABASI_ALBERT", 70, 0.442857, "Barabási-Albert"),
    ("GraphType.BARABASI_ALBERT", 60, 0.483333, "Barabási-Albert"),
    ("GraphType.BARABASI_ALBERT", 50, 0.400000, "Barabási-Albert"),
    ("GraphType.BARABASI_ALBERT", 40, 0.450000, "Barabási-Albert"),
    ("GraphType.BARABASI_ALBERT", 30, 0.366667, "Barabási-Albert"),
    ("GraphType.BARABASI_ALBERT", 20, 0.300000, "Barabási-Albert"),
    ("TREE", 10, 0.900000, "Tree"),
    ("TREE", 50, 0.780000, "Tree"),
    ("TREE", 100, 0.700000, "Tree"),
    ("TREE", 500, 0.646000, "Tree"),
    ("STAR", 20, 0.100000, "Star"),
    ("STAR", 100, 0.020000, "Star"),
    ("STAR", 200, 0.010000, "Star"),
    ("STAR", 2000, 0.001000, "Star"),
    ("ERDOS RENYI", 2000, 1.000000, "Erdős–Rényi"),
    ("ERDOS RENYI", 1000, 1.000000, "Erdős–Rényi"),
    ("ERDOS RENYI", 200, 1.000000, "Erdős–Rényi"),
    ("ERDOS RENYI", 100, 1.000000, "Erdős–Rényi"),
    ("ERDOS RENYI", 20, 1.000000, "Erdős–Rényi"),
    ("TRIANGULAR", 18, 0.333000, "Triangular"),
    ("TRIANGULAR", 24, 0.500000, "Triangular"),
    ("TRIANGULAR", 176, 0.500000, "Triangular"),
    ("TRIANGULAR", 284, 0.271000, "Triangular"),
    ("TRIANGULAR", 546, 0.500000, "Triangular"),
    ("SQUARE GRID", 40000, 0.126000, "Square Grid"),
    ("SQUARE GRID", 10000, 0.128000, "Square Grid"),
    ("SQUARE GRID", 25, 0.240000, "Square Grid"),
    ("SQUARE GRID", 16, 0.188000, "Square Grid"),
    ("SQUARE GRID", 16, 0.188000, "Square Grid"),
]

real = [    
    ("collab", 235868, 0.670000, "Real"),
    ("ddi", 4267, 0.920000, "Real"),
    ("ppa", 576289, 0.990000, "Real"),
    ("citation2", 2927963, 0.620000, "Real"),
    ("Pubmed", 19717, 0.660000, "Real"),
    ("Cora", 2708, 0.870000, "Real"),
    ("Citeseer", 3327, 0.630000, "Real"),
    ("Photo", 7650, 0.970000, "Real"),
    ("Computers", 13752, 0.960000, "Real")
    ]

TITLE_SIZE = 26
LABEL_SIZE = 35
TICK_SIZE = 35
LEGEND_SIZE = 20
LEGEND_TITLE_SIZE = 18
ANNOTATION_SIZE = 24
FIGSIZE = (10, 8)
DPI = 300
LEGENG_SIZE = 15

df = pd.DataFrame(data, columns=["Dataset", "NumNodes", "AutoV", "GraphType"])

# 调色板
palette = {
    "Real": "#1f77b4",
    "Barabási-Albert": "#ff7f0e",
    "Tree": "#2ca02c",
    "Star": "#d62728",
    "Erdős–Rényi": "#9467bd",
    "Triangular": "#8c564b",
    "Square Grid": "#e377c2"
}

# 绘图
fig = plt.figure(figsize=FIGSIZE)  # 
gs = GridSpec(3, 3)  # 

ax_main = fig.add_subplot(gs[0:3, 0:3])

# 散点图
sns.scatterplot(
    data=df,
    x="NumNodes",
    y="AutoV",
    hue="GraphType",
    palette=palette,
    edgecolor=None,  # 去掉黑边框
    linewidth=2,  # 散点加粗
    s=60,
    alpha=0.9,
    marker="X",  # 使用 "X" 作为标记
    ax=ax_main
)
real = pd.DataFrame(real, columns=["Dataset", "NumNodes", "AutoV", "GraphType"])


sns.scatterplot(
    data=real,
    x="NumNodes",
    y="AutoV",
    hue="GraphType",
    palette=palette,
    edgecolor=None,   # Remove black borders
    linewidth=0,      # No outline needed for stars
    s=300,            # Increase marker size (e.g., 300)
    alpha=0.9,
    marker="*",       # "*" is the five-pointed star marker
    ax=ax_main
)
# 手动调整标注位置，避免重叠，确保不同点的标注在上方或右侧
# real_df = real[df["GraphType"] == "Real"]
# a 是你想要平移的量，比如 a = 0.02
a = 0.974
b = -0.05
positions = {
    "Photo": (0.1 + a, -0.1 - b),
    "Computers": (0.05 + a, -0.15 - b),
    "Cora": (0.05 + a, -0.15 - b),
    "Citeseer": (0.05, 0.02),
    "Pubmed": (0.05, 0.02),
    "collab": (0.05, 0.02),
    "ddi": (0.05 + a, -0.15 - b),
    "ppa": (0.05, 0.02),
    "citation2": (0.05, 0.02),
}

for _, row in real.iterrows():
    x, y = row["NumNodes"], row["AutoV"]
    label = row["Dataset"]
    
    # 根据预定义的位置进行标注
    if label in positions:
        offset_x, offset_y = positions[label]
    else:
        offset_x, offset_y = 0, 0  # 默认位置
    
    ax_main.text(
        x + offset_x, y + offset_y, label,
        fontsize=20, color='black', ha='center', va='bottom'  # 放大字体
    )



# 主图设置，进一步放大字体
ax_main.set_xscale("log")
ax_main.set_xlim(1e0, 1e7)
y_min, y_max = 0, 1
y_margin = (y_max - y_min) * 0.1  # 10% margin
ax_main.set_ylim(y_min - y_margin, y_max + y_margin)
ax_main.tick_params(axis='both', labelsize=LABEL_SIZE)
ax_main.set_xlabel("Number of Nodes (log scale)", fontsize=LABEL_SIZE)  # 
ax_main.set_ylabel(r"Automorphism Ratio $\alpha_{\mathcal{V}}$", fontsize=LABEL_SIZE)  # 
# ax_main.set_title("Automorphism Spectrum of Graphs", fontsize=TITLE_SIZE)  # 
ax_main.grid(True, linestyle="--", linewidth=0.5)

# 分类标记框 - 右下角，横坐标上方

ax_main.legend(title="Graph Type", fontsize=LEGEND_SIZE, title_fontsize=LEGEND_SIZE, loc="lower right", frameon=True, fancybox=True, framealpha=0.5)

# 布局调整与保存
plt.tight_layout(pad=1.0)  # 防止标注溢出
output_path = "/hkfs/work/workspace/scratch/cc7738-rebuttal/ANP4Link/syn_real/results/dist.pdf"
plt.savefig(output_path, dpi=DPI)
plt.show()

output_path






