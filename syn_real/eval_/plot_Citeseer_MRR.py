data = """
Metric,Hits@1,MRR,AUC,AP
Citeseer_GCN_inter0.00_intra0.00_total0_Orbits_658.00_Norm_0.81_ArScore_0.90,26.52 ± 14.78,48.61 ± 13.29,99.65 ± 0.08,99.59 ± 0.11
"""

import seaborn as sns
models = ["GCN", "GAT", "GIN", "GraphSAGE", "MixHopGCN", "ChebGCN", "LINKX"]
set3_colors = sns.color_palette("Set3", len(models))

# Assign colors to models
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict
from plot_Citeseer_AUC import model_colors


raw_data = [
    ("GCN",
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
     [49.84, 48.61, 48.34, 43.7, 39.74, 27.99, 21.32, 20.78, 18.08, 13.18, 10.56],
     [4.78, 13.29, 10.77, 10.66, 9.66, 7.7, 4.65, 3.08, 3.87, 2.0, 2.93]),
    # ("GAT",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [31.93, 27.72, 25.82, 24.61, 23.06, 21.82, 21.16, 19.1, 14.75, 13.61, 10.22],
    #  [8.23, 7.61, 4.77, 6.22, 3.3, 8.21, 4.89, 4.48, 2.74, 5.15, 1.75]),
    # ("GIN",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [20.37, 18.09, 15.77, 14.17, 11.37, 10.79, 10.45, 9.61, 7.51, 5.91, 2.82],
    #  [3.99, 4.48, 2.23, 3.56, 2.29, 2.36, 1.82, 3.08, 3.36, 2.72, 2.89]),
    # ("GraphSAGE",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [24.01, 20.28, 20.13, 16.45, 15.82, 12.57, 10.61, 10.56, 10.34, 6.89, 5.16],
    #  [15.47, 12.22, 10.96, 4.91, 5.18, 5.13, 5.2, 5.06, 3.02, 2.12, 2.18]),
    # ("LINKX",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [61.82, 56.8, 51.21, 50.44, 44.59, 39.72, 36.21, 32.68, 32.09, 18.6, 12.34],
    #  [5.53, 7.94, 6.15, 15.06, 15.02, 9.74, 4.3, 8.6, 12.39, 2.41, 4.56]),
    # ("MixHopGCN",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [70.99, 52.17, 51.84, 43.61, 39.02, 36.87, 36.12, 35.69, 34.84, 25.08, 14.32],
    #  [27.71, 4.79, 9.3, 6.49, 2.52, 10.14, 4.8, 9.47, 6.12, 11.08, 2.42]),

    ("BUDDY",
    [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
    [29.06, 31.40, 26.07, 37.07, 40.97, 38.80, 22.80, 37.20, 32.22, 35.62, 34.86],
    [7.08, 4.33, 8.50, 5.59, 4.09, 1.98, 4.00, 3.34, 3.14, 5.89, 5.28]),
    # # ("ChebGCN",
    # #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    # #  [28.4, 27.45, 27.07, 23.05, 22.76, 15.23, 12.33, 8.83, 7.57, 7.09, 4.81],
    # #  [6.49, 5.14, 10.19, 3.27, 4.13, 4.23, 3.0, 2.23, 2.49, 0.94, 0.62]),
     ("GCN-LAP",
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
     [63.75, 64.01, 74.99, 60.41, 49.12, 30.90, 37.15, 38.19, 24.42, 26.08, 23.36],
     [8.76, 8.85, 8.55, 9.69, 7.91, 4.64, 6.31, 6.80, 2.50, 4.46, 4.97]),

    ("GCN-DW",
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
     [65.92, 53.50, 75.96, 61.99, 50.64, 32.26, 37.06, 35.14, 26.22, 29.03, 25.11],
     [9.59, 9.29, 10.76, 8.48, 10.31, 3.79, 4.30, 4.28, 2.46, 3.65, 5.44]),

     ("GCN-RF",
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
     [60.49, 49.02, 59.79, 58.83, 59.68, 65.63, 54.62, 30.14, 34.67, 41.84, 18.56],
     [9.57, 12.14, 12.07, 11.46, 11.20, 13.42, 9.97, 6.63, 4.98, 5.38, 3.56]),

    # ("Proposed w.o. D1", # 22.36
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [83.4, 71.65, 69.35, 67.94, 64.67, 64.18, 63.9, 43.61, 41.19, 41.19, 36.42],
    #  [11.4, 12.54, 12.85, 12.46, 8.23, 7.58, 9.91, 5.35, 3.23, 3.23, 9.45]),

    # ("Proposed w.o D2", # -10.09
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [79.99, 76.28, 72.2, 66.62, 66.62, 66.62, 58.34, 55.88, 49.56, 41.81, 38.42],
    #  [12.23, 7.76, 16.27, 12.13, 12.13, 12.13, 13.8, 12.07, 10.0, 7.54, 12.32]),

    # ("Proposed w.o Dropout",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [71.82, 66.8, 61.21, 60.44, 54.59, 49.72, 46.21, 42.68, 42.09, 28.6, 22.34],
    #  [12.23, 7.76, 16.27, 12.13, 12.13, 12.13, 13.8, 12.07, 10.0, 7.54, 12.32]),

    ("EO-GNN",
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
     [85.72, 83.02, 74.85, 72.06, 68.64, 67.42, 66.61, 65.97, 55.19,  42.59, 41.19],
     [5.57, 6.87, 7.6, 5.95, 4.59, 10.02, 16.27, 11.65, 12.1, 8.04, 3.23]),
]


new_alpha = np.arange(0.1, 1.0, 0.2)
interpolated_data = defaultdict(dict)

LABEL_SIZE = 28
TICK_SIZE = 28
LEGEND_SIZE = 24
FIGSIZE = (10, 8)
DPI = 300


# Perform interpolation for each model
for model, alpha, best_valid, variance in raw_data:
    interpolated_data[model]["alpha"] =  [1-i for i in alpha] 
    interpolated_data[model]["best_valid"] = best_valid
    interpolated_data[model]["variance"] = variance
    
fig, ax = plt.subplots(figsize=(10, 6))

dashed_models = {"ChebGCN", "LINKX", "GIN"}  # Models that will have dashed lines
line_styles = {model: "--" if model in dashed_models else "-" for model in interpolated_data.keys()}
alpha_values = {model: 0.5 if model in dashed_models else 1.0 for model in interpolated_data.keys()}  # Reduce opacity for dashed lines



model_colors["GAT"] = "pink"
model_colors["EO-GNN"] = "red"
# Plot interpolated data with solid markers and error bars
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
        fmt='o',
        color=color,
        alpha=0.3, 
        capsize=6,
        elinewidth=2,
        capthick=2
    )

# Legend
legend = ax.legend(
    fontsize=LEGEND_SIZE,
    loc="upper right",
    bbox_to_anchor=(0.95, 1),
    ncol=2,
    frameon=True,
    framealpha=0.5,
    fancybox=True
)
legend.get_frame().set_facecolor('white')

ax.set_xlabel(r"$EAR$", fontsize=LABEL_SIZE)
ax.set_ylabel("MRR (/%)", fontsize=LABEL_SIZE)
ax.set_xticks(new_alpha)
ax.set_yticks(np.arange(0, 101, 20))
ax.tick_params(axis='both', labelsize=TICK_SIZE)
ax.set_ylim(10, 100)
plt.tight_layout()
base = '/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/eval_'
plt.savefig(f"{base}/new_Exp1_Citeseer_SYN_Real_MRR.pdf")