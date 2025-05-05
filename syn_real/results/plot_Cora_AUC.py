import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict

raw_data = [
    ("GCN",
     [0.81, 0.70, 0.63, 0.49, 0.40, 0.29, 0.19, 0.17, 0.10],
     [99.41, 99.39, 99.06, 98.51, 98.01, 97.27, 96.45, 96.29, 94.07],
     [0.23, 0.09, 0.22, 0.35, 0.45, 0.42, 0.57, 0.67, 0.86]),

    ("GAT",
     [0.81, 0.70, 0.63, 0.49, 0.40, 0.29, 0.19, 0.17, 0.10],
     [99.28, 98.97, 98.99, 98.75, 98.48, 97.98, 97.72, 97.51, 96.42],
     [0.16, 0.23, 0.13, 0.20, 0.25, 0.22, 0.50, 0.50, 0.39]),

    ("GIN",
     [0.81, 0.70, 0.63, 0.49, 0.40, 0.29, 0.19, 0.17, 0.10],
     [91.83, 95.76, 95.53, 94.91, 84.81, 82.49, 70.24, 72.01, 65.24],
     [0.70, 0.78, 0.59, 0.90, 0.14, 0.14, 0.78, 0.96, 0.38]),

    ("GraphSAGE",
     [0.81, 0.70, 0.63, 0.49, 0.40, 0.29, 0.19, 0.17, 0.10],
     [97.91, 97.96, 97.96, 97.05, 95.93, 93.38, 90.27, 90.37, 85.82],
     [0.62, 0.57, 0.60, 0.85, 0.93, 0.99, 1.93, 2.24, 2.03]),

    ("MixHopGCN",
     [0.81, 0.70, 0.63, 0.49, 0.40, 0.29, 0.19, 0.17, 0.10],
     [99.55, 99.28, 98.99, 98.75, 98.48, 97.98, 97.72, 97.51, 96.42],
     [0.17, 0.09, 0.13, 0.20, 0.25, 0.22, 0.50, 0.50, 0.39]),

    ("ChebGCN",
     [0.81, 0.70, 0.63, 0.50, 0.49, 0.40, 0.29, 0.19, 0.17, 0.10],
     [98.41, 98.19, 97.82, 96.56, 96.76, 94.09, 91.18, 86.24, 84.88, 77.29],
     [0.13, 0.22, 0.21, 0.79, 0.32, 0.56, 1.02, 1.77, 1.42, 3.20]),

    ("LINKX",
     [0.81,  0.69,  0.62,  0.49,  0.41, 0.30, 0.20, 0.18, 0.11],
     [97.30, 96.99, 96.81, 94.28, 92.82, 89.63, 86.69, 85.47, 83.48],
     [0.32, 0.15, 0.18, 0.15,  0.13,  0.29, 0.15, 0.12, 0.18]),

    ("Proposed w.o. D1",
     [0.81, 0.7, 0.63,  0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
     [99.89, 99.83, 99.77, 99.76, 99.76, 99.74, 99.67, 99.63, 99.62],
     [0.08, 0.1, 0.11, 0.07, 0.05, 0.07, 0.12, 0.07, 0.22]),
    ("Proposed w.o D2",
     [0.81, 0.7, 0.63,  0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
     [99.9, 99.8, 99.79, 99.78, 99.75, 99.74, 99.73, 99.71, 99.9],
     [0.05, 0.05, 0.07, 0.12, 0.03, 0.07, 0.04, 0.03, 0.05]),
    ("Proposed",
     [0.81, 0.7, 0.63,  0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
     [99.94, 99.86, 99.83, 99.81, 99.8, 99.78, 99.76, 99.74, 99.73],
     [0.03, 0.03, 0.07, 0.0, 0.05, 0.04, 0.03, 0.07, 0.08]),
]

import seaborn as sns
models = ["GCN", "GAT", "GIN", "GraphSAGE", "MixHopGCN", "ChebGCN", "LINKX"]
set3_colors = sns.color_palette("Set3", len(models))
model_colors = {model: color for model, color in zip(models, set3_colors)}
new_alpha = np.arange(0.1, 1.0, 0.1)
interpolated_data = defaultdict(dict)
for model, alpha, best_valid, variance in raw_data:

    interpolated_data[model]["alpha"] =  [1-i for i in alpha] 
    interpolated_data[model]["best_valid"] = best_valid
    interpolated_data[model]["variance"] = variance

fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.cm.get_cmap('tab10', len(interpolated_data))
dashed_models = {"ChebGCN", "LINKX", "GIN"}  # Models that will have dashed lines
line_styles = {model: "--" if model in dashed_models else "-" for model in interpolated_data.keys()}
alpha_values = {model: 0.5 if model in dashed_models else 1.0 for model in interpolated_data.keys()}  # Reduce opacity for dashed lines
model_colors["GAT"] = "pink"
for idx, (model, values) in enumerate(interpolated_data.items()):
    color = model_colors.get(model, f"C{idx}")  # consistent color fallback

    # Plot main lines
    ax.plot(
        values["alpha"],
        values["best_valid"],
        linestyle=line_styles[model],
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
        fmt='o',  # Markers only for error bars
        color=color,
        alpha=0.3,  # Reduced transparency for error bars
        capsize=6,
        elinewidth=2,
        capthick=2
    )
    
fontsize = 22
# Formatting the plot
ax.set_xlabel(r"$\alpha$", fontsize=fontsize)
ax.set_ylabel("AUC (/%)", fontsize=fontsize)
ax.set_xticks(new_alpha)
ax.set_yticks(np.arange(60, 101, 10))
ax.tick_params(axis='both', labelsize=fontsize) 
fontsize = 16
ax.legend(fontsize=fontsize, loc="lower left")
plt.tight_layout()

plt.savefig('Cora_SYN_Real.pdf')
