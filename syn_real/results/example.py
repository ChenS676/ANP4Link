import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 模拟你的 raw_data
raw_data = [
    ("GCN",
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
     [99.65, 99.50, 99.42, 98.92, 98.61, 98.03, 97.65, 96.32, 96.49, 95.03, 91.81],
     [0.08, 0.14, 0.16, 0.24, 0.33, 0.22, 0.47, 0.56, 0.74, 1.60, 2.05]),

    # ("LINKX",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
    #  [97.30, 96.99, 96.81, 94.28, 92.82, 89.63, 86.69, 85.47, 83.48, 81.28, 79.68],
    #  [0.08, 0.14, 0.16, 0.24, 0.33, 0.22, 0.47, 0.56, 0.74, 1.42, 1.49]),

    ("BUDDY",
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
     [95.2, 95.83, 95.84, 94.37, 95.1, 95.79, 96.03, 94.86, 94.7, 95.5, 95.57],
     [0.39, 0.18, 0.26, 0.53, 0.51, 0.48, 0.32, 0.42, 0.31, 0.39, 0.41]),


    ('Proposed w.o. D1',
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
     [99.89, 99.83, 99.77, 99.76, 99.76, 99.74, 99.67, 99.63, 99.625, 99.6225, 99.62],
     [0.08, 0.1, 0.11, 0.07, 0.05, 0.07, 0.12, 0.07, 0.145, 0.1825, 0.22]),

    ('Proposed w.o D2',
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
     [99.9, 99.8, 99.79, 99.78, 99.75, 99.74, 99.73, 99.71, 99.7775, 99.83875, 99.9],
     [0.05, 0.05, 0.07, 0.12, 0.03, 0.07, 0.04, 0.03, 0.04, 0.045, 0.05]),

    ('Proposed',
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
     [99.94, 99.86, 99.83, 99.81, 99.8, 99.78, 99.76, 99.74, 99.735, 99.7325, 99.73],
     [0.03, 0.03, 0.07, 0.0, 0.05, 0.04, 0.03, 0.07, 0.075, 0.0775, 0.08])
]

# 新的x轴
new_alpha = np.arange(0.1, 1.0, 0.2)

# 存储
interpolated_data = defaultdict(dict)

for model, alpha, best_valid, variance in raw_data:
    interpolated_data[model]["alpha"] = [1-i for i in alpha]
    interpolated_data[model]["best_valid"] = best_valid
    interpolated_data[model]["variance"] = variance

# 画图
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

# 定义哪些模型用左轴、哪些用右轴
left_models = ['Proposed', 'Proposed w.o D2', 'Proposed w.o. D1']
right_models = ['GCN', 'LINKX']

# 模型颜色
model_colors = {
    'GCN': 'tab:blue',
    'LINKX': 'tab:green',
    'Proposed': 'red',
    'Proposed w.o D2': 'darkred',
    'Proposed w.o. D1': 'firebrick'
}

# 线型设定
line_styles = {
    'GCN': '--',
    'LINKX': '--',
    'Proposed': '-',
    'Proposed w.o D2': '-.',
    'Proposed w.o. D1': ':'
}

# 绘制
for model, values in interpolated_data.items():
    color = model_colors.get(model, 'black')
    linestyle = line_styles.get(model, '-')

    if model in left_models:
        ax = ax1
    else:
        ax = ax2

    ax.plot(
        values["alpha"],
        values["best_valid"],
        linestyle=linestyle,
        linewidth=2,
        color=color,
        label=model,
        marker='o',
        markersize=6,
        markerfacecolor=color,
        markeredgecolor='black',
        markeredgewidth=0.8
    )
    ax.errorbar(
        values["alpha"],
        values["best_valid"],
        yerr=values["variance"],
        fmt='o',
        color=color,
        alpha=0.3,
        capsize=6,
        elinewidth=2,
        capthick=2
    )

# 设置标签
LABEL_SIZE = 30
TICK_SIZE = 28
LEGEND_SIZE = 20
LEGEND_TITLE_SIZE = 24

ax1.set_xlabel(r"$\alpha_{\mathcal{E}}$", fontsize=LABEL_SIZE)
ax1.set_ylabel("AUC (%) - Proposed", fontsize=LABEL_SIZE, color='red')
ax2.set_ylabel("AUC (%) - Baselines", fontsize=LABEL_SIZE, color='tab:blue')

ax1.set_xticks(new_alpha)
ax1.tick_params(axis='both', labelsize=TICK_SIZE)
ax2.tick_params(axis='both', labelsize=TICK_SIZE)

# 合并 legend
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=LEGEND_SIZE, loc="lower left", frameon=False)

plt.tight_layout()
plt.savefig('Citeseer_SYN_Real_twinaxis.pdf', dpi=300)
plt.show()
