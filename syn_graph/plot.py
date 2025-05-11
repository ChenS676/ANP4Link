import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import multivariate_normal

# 假设你已有 df_viz_cleaned 和 df_real
# 包含字段：AutoV（自动同构率）、NumNodes（节点数量）、Dataset（图名）、GraphType（图类型）


import pandas as pd

data = [
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


# 配色和标记设置
color_map = {
    "Barabási-Albert": "#e41a1c",
    "Tree": "#377eb8",
    "Triangular": "#4daf4a",
    "Square Grid": "#984ea3",
    "Star": "#ff7f00",
    "Erdős–Rényi": "#a65628",
    "Real": "#999999"
}
marker_map = {
    "Barabási-Albert": "o",
    "Tree": "s",
    "Triangular": "D",
    "Square Grid": "^",
    "Star": "P",
    "Erdős–Rényi": "X",
    "Real": "h"
}
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
csv_file = '/hkfs/work/workspace/scratch/cc7738-rebuttal/ANP4Link/syn_graph/summary.csv'
syn_data = pd.read_csv(csv_file)

df_viz_cleaned = pd.DataFrame(syn_data, columns=["data_name", "num_nodes", "automorphism_score"])
df_real = data

# 参数区：你可以随时调整这些值
TITLE_SIZE = 26
LABEL_SIZE = 24
TICK_SIZE = 24
LEGEND_SIZE = 24
LEGEND_TITLE_SIZE = 24
ANNOTATION_SIZE = 24
FIGSIZE = (10, 10)
DPI = 300

# 绘图开始
plt.figure(figsize=FIGSIZE)

# 每个类别的 KDE 密度图（x 为节点数，y 为自动同构率）
for graph_type in df_viz_cleaned["GraphType"].unique():
    color = color_map[graph_type]
    cmap = sns.light_palette(color, as_cmap=True)
    
    sns.kdeplot(
        data=df_viz_cleaned[df_viz_cleaned["GraphType"] == graph_type],
        x="NumNodes",
        y="AutoV",
        fill=True,
        cmap=cmap,
        levels=100,
        thresh=0.01,
        alpha=0.45
    )

# 可调字体和样式设置
TITLE_SIZE = 18
LABEL_SIZE = 18
TICK_SIZE = 18
LEGEND_SIZE = 11
LEGEND_TITLE_SIZE = 18
ANNOTATION_SIZE = 18
FIGSIZE = (10, 10)
DPI = 300

color_map = {
    "Barabási-Albert": "#e41a1c",
    "Tree": "#377eb8",
    "Triangular": "#4daf4a",
    "Square Grid": "#984ea3",
    "Star": "#ff7f00",
    "Erdős–Rényi": "#a65628",
    "Real": "#999999"
}
marker_map = {
    "Barabási-Albert": "o",
    "Tree": "s",
    "Triangular": "D",
    "Square Grid": "^",
    "Star": "P",
    "Erdős–Rényi": "X",
    "Real": "h"
}

# 绘图
plt.figure(figsize=FIGSIZE)

# 背景密度图（每类图）
for graph_type in df_viz_cleaned["GraphType"].unique():
    color = color_map[graph_type]
    cmap = sns.light_palette(color, as_cmap=True)
    sns.kdeplot(
        data=df_viz_cleaned[df_viz_cleaned["GraphType"] == graph_type],
        x="NumNodes",
        y="AutoV",
        fill=True,
        cmap=cmap,
        levels=100,
        thresh=0.01,
        alpha=0.45
    )

# 绘制数据点
for graph_type in df_viz_cleaned["GraphType"].unique():
    subset = df_viz_cleaned[df_viz_cleaned["GraphType"] == graph_type]
    size = 150 if graph_type == "Real" else 70  # 放大真实数据点
    plt.scatter(
        subset["NumNodes"],
        subset["AutoV"],
        label=graph_type,
        c=color_map[graph_type],
        marker=marker_map[graph_type],
        s=size,
        edgecolors="black",
        linewidth=0.3,
        alpha=0.85
    )

# 标注真实数据图名
for _, row in df_real.iterrows():
    plt.text(
        row["NumNodes"],
        row["AutoV"] + 0.01,
        row["Dataset"],
        fontsize=ANNOTATION_SIZE + 1,
        horizontalalignment='center'
    )

# 图形设置
plt.xscale("log")
plt.xlabel("Number of Nodes (log scale)", fontsize=LABEL_SIZE)
plt.ylabel(r"Automorphism Ratio $\alpha_{\mathcal{V}}$", fontsize=LABEL_SIZE)
plt.title("Automorphism Spectrum of Graphs", fontsize=TITLE_SIZE)
plt.xticks(fontsize=TICK_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.grid(True, linestyle="--", linewidth=0.5)

# 图例置于图内右上角，透明度降低
plt.legend(
    title="Graph Type",
    fontsize=LEGEND_SIZE,
    title_fontsize=LEGEND_TITLE_SIZE,
    loc="upper right",
    frameon=True,
    fancybox=True,
    framealpha=0.6,
    borderpad=0.5
)

# 保存 + 展示
plt.tight_layout()
plt.savefig("graph_type_density_plot.pdf", dpi=DPI)
plt.show()

