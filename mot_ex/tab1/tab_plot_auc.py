import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

# === Define full raw_data with α = 0.6 ===
raw_data = [
    ("GCN", [0.125, 0.25,  0.375, 0.5,   0.625, 0.75, 0.875, 1.0],
            [79.09, 83.79, 86.65, 86.1,  89.94, 91.76, 94.51, 96.09],
            [0.89, 1.89, 1.87, 2.72, 1.56, 1.62, 0.77, 0.43]),

    ("GAT", [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            [92.89, 95.24, 95.43, 95.85, 97.48, 97.88, 99.05, 99.38],
            [0.17, 0.49, 0.24, 0.21, 0.22, 0.17, 0.07, 0.15]),

    ("SAGE", [0.125, 0.25,  0.375, 0.5,   0.625, 0.75, 0.875, 1.0],
            [79.13, 84.92, 85.12, 89.06, 93.7, 94.88, 97.62, 98.57],
            [2.08, 1.24, 0.99, 1.09, 0.56, 0.83, 0.13, 0.27]),

    ("ChebGCN", [0.125, 0.25,  0.375, 0.5,   0.625, 0.75, 0.875, 1.0],
                [97.16, 97.11, 97.96, 97.59, 98.55, 97.89, 97.55, 97.88],
                [0.25, 0.29, 0.06, 0.05, 0.27, 0.44, 0.49, 0.46]),

    ("LINKX",  [0.125, 0.25,  0.375, 0.5,   0.625, 0.75, 0.875, 1.0],
                [93.12, 93.5, 94.53, 94.06, 94.17, 94.93, 94.87, 94.39],
                [0.55, 0.53, 0.06, 0.22, 0.24, 0.42, 0.38,  0.53]),
    
    ("MixHopGCN", [0.125, 0.25,  0.375, 0.5,   0.625, 0.75, 0.875, 1.0],
                [50.0, 50.0, 83.0, 90.01, 97.03, 98.06, 99.11, 99.58],
                [0.0, 0.0, 0.27, 0.31, 0.27, 0.31, 0.17, 0.1]),

    # ("Proposed w.o. D1", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    #  [84.35, 87.27, 88.44, 91.27, 87.92, 84.58, 92.2, 100.0, 98.22, 99.04, 98.61],
    #  [30.61, 28.18, 27.25, 24.89, 26.51, 28.13, 13.93, 0.16, 14.85, 29.87, 44.76]),

    # ("Proposed w.o D2", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    #  [84.62, 87.85, 91.47, 100.0, 96.06, 92.11, 96.04, 99.08, 99.62, 100.0, 99.74],
    #  [43.23, 28.35, 13.6, 0.0, 14.42, 28.84, 15.32, 0.07, 7.71, 14.18, 21.05]),

    # ("Proposed", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    #  [98.55, 98.91, 100.0, 98.77, 99.38, 100.0, 98.65, 100.0, 100.0, 99.77, 99.94],
    #  [0.0, 0.22, 0.0, 0.0, 0.0, 0.0, 0.09, 0.79, 0.08, 0.0, 0.14]),
]

# === Prepare and plot ===
plot_data = defaultdict(dict)
for model, alpha, best_valid, variance in raw_data:
    plot_data[model]["alpha"] = [round(1 - a, 2) for a in alpha]
    plot_data[model]["best_valid"] = best_valid
    plot_data[model]["variance"] = variance

baselines = ["GCN", "GAT", "GIN", "GraphSAGE", "MixHopGCN", "ChebGCN", "LINKX"]
proposed = ["Proposed w.o. D1", "Proposed w.o D2", "Proposed"]

def is_yellow(rgb): return rgb[0] > 0.9 and rgb[1] > 0.9 and rgb[2] < 0.6
palette = sns.color_palette("Set2", len(baselines))
baseline_colors = [c for c in palette if not is_yellow(c)][:len(baselines)]
model_colors = {m: c for m, c in zip(baselines, baseline_colors)}
proposed_colors = [(0.1, 0.3, 0.6), (0.2, 0.5, 0.2), (0.6, 0.1, 0.2)]
model_colors.update({m: c for m, c in zip(proposed, proposed_colors)})

dashed_models = {"ChebGCN", "LINKX"}
line_styles = {m: "--" if m in dashed_models else "-" for m in plot_data}

fig, ax = plt.subplots(figsize=(10, 8))

for idx, (model, values) in enumerate(plot_data.items()):
    color = model_colors.get(model, f"C{idx}")
    if model == 'LINKX':
        color = 'red'
    alphas = values["alpha"]
    scores = values["best_valid"]
    variances = values["variance"]

    ax.plot(
        alphas, scores,
        linestyle=line_styles.get(model, "-"),
        linewidth=2.2, color=color,
        label=model, marker='o',
        markersize=5.5, markerfacecolor=color,
        markeredgecolor='black', markeredgewidth=0.6
    )

    lower_var = np.array(variances) * 0.8
    upper_var = np.array(variances) * 0.2

    ax.errorbar(
        alphas, scores, yerr=[lower_var, upper_var],
        fmt='o', color=color, alpha=0.25,
        capsize=4, elinewidth=1.4, capthick=1.4
    )

fontsize = 40
ax.set_xlabel(r"$\alpha$", fontsize=fontsize)
ax.set_ylabel("AUC (/%)", fontsize=fontsize)
ax.set_xticks(sorted(set(alphas)))
ax.set_yticks(np.arange(80, 110, 5))
ax.set_ylim(78, 100)  # <-- This line limits the y-axis range
ax.tick_params(axis='x', labelsize=25)  # 横轴刻度字体大小
ax.tick_params(axis='y', labelsize=30)  # 纵轴刻度字体大小

ax.legend(fontsize=30, loc="lower left", frameon=False, ncol=1)
plt.tight_layout()
plt.savefig("Tri_SYN_Real2_ALL.pdf", bbox_inches='tight')
plt.show()