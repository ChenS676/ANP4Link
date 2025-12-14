
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

# === Configuration ===
# TITLE_SIZE = 26
# LABEL_SIZE = 24
# TICK_SIZE = 24
# LEGEND_SIZE = 24
# LEGEND_TITLE_SIZE = 24
# ANNOTATION_SIZE = 24
# FIGSIZE = (10, 8)
# DPI = 300

TITLE_SIZE = 26
LABEL_SIZE = 35
TICK_SIZE = 35
LEGEND_SIZE = 26
LEGEND_TITLE_SIZE = 18
ANNOTATION_SIZE = 24
FIGSIZE = (10, 8)
DPI = 300
LEGENG_SIZE = 15

# === Prepare and plot ===
plot_data = defaultdict(dict)
for model, alpha, best_valid, variance in raw_data:
    plot_data[model]["alpha"] = alpha
    plot_data[model]["best_valid"] = best_valid
    plot_data[model]["variance"] = variance

baselines = ["GCN", "GAT", "GIN", "SAGE", "MixHopGCN", "ChebGCN", "LINKX"]
proposed = ["Proposed"]

def is_yellow(rgb): return rgb[0] > 0.9 and rgb[1] > 0.9 and rgb[2] < 0.6
palette = sns.color_palette("Set2", len(baselines))
baseline_colors = [c for c in palette if not is_yellow(c)][:len(baselines)]
model_colors = {m: c for m, c in zip(baselines, baseline_colors)}
proposed_colors = [(0.1, 0.3, 0.6), (0.2, 0.5, 0.2), (0.6, 0.1, 0.2)]
model_colors.update({m: c for m, c in zip(proposed, proposed_colors)})

dashed_models = {"ChebGCN", "LINKX", "GIN"}
line_styles = {m: "--" if m in dashed_models else "-" for m in plot_data}

fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

for idx, (model, values) in enumerate(plot_data.items()):
    color = model_colors.get(model, f"C{idx}")
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
new_alpha = np.arange(0.1, 1.0, 0.2)
ax.set_xlabel(r"$\alpha_\mathcal{E}$", fontsize=LABEL_SIZE)
ax.set_ylabel("AUC (/%)", fontsize=LABEL_SIZE)

ax.set_xticks(new_alpha)
ax.set_yticks(np.arange(80, 110, 20))
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.legend(fontsize=LEGEND_SIZE, loc="lower left", frameon=False, ncol=1)
plt.tight_layout()
plt.savefig("Tri_SYN_Real2_ALL.pdf", bbox_inches='tight')
plt.show()
