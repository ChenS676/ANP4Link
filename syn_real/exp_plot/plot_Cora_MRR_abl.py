import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict
from syn_real.exp_plot.plot_Citeseer_AUC_abl import model_colors


raw_cora_mrr_data = [
    # ("GCN",
    #  [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
    #  [42.28, 34.09, 27.06, 21.29, 20.83, 18.37, 16.5, 14.78, 12.82],
    #  [11.15, 8.75, 5.83, 6.17, 3.29, 3.9, 3.5, 4.48, 3.91]),
    
    # ("GAT",
    #  [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
    #  [30.45, 25.24, 21.77, 20.28, 19.74, 17.06, 15.6, 14.8, 13.28],
    #  [6.74, 7.63, 7.0, 5.6, 9.83, 6.54, 3.41, 4.85, 3.1]),
    
    # ("GCN-LAP",
    #  [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
    #  [34.06, 34.24, 23.08, 33.95, 37.39, 29.56, 30.49, 33.04, 15.52],
    #  [9.46, 9.45, 5.45, 7.70, 7.80, 2.86, 4.94, 7.21, 2.65]),

    #  ("GCN-DW",
    #  [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
    # [72.87, 43.98, 29.52, 40.14, 53.73, 35.98, 42.41, 50.71, 21.16],
    # [12.57, 7.79, 4.48, 5.64, 11.90, 3.61, 6.86, 9.77, 3.57]),

    # ("GCN-RF",
    #  [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
    #  [52.38, 48.97, 44.79, 61.81, 27.26, 29.51, 44.57, 24.89, 34.98],
    #  [13.90, 6.06, 10.38, 9.20, 3.79, 7.91, 11.12, 4.14, 10.87]),

    # ("BUDDY",
    # [0.81,  0.69,  0.62,  0.49,  0.41, 0.30, 0.20, 0.18, 0.11],
    #  [33.41, 23.80, 32.42, 23.09, 19.55, 27.87, 31.36, 27.51, 18.71],
    #  [5.10, 3.71, 1.96, 3.11, 1.05, 7.83, 4.70, 1.46, 1.04]),

    # ("GIN",
    #  [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
    #  [15.8, 15.57, 14.43, 10.88, 9.19, 6.87, 4.42, 4.06, 3.07],
    #  [2.59, 5.25, 3.99, 4.81, 3.93, 4.0, 3.89, 3.72, 3.3]),
    
    # ("GraphSAGE",
    #  [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
    #  [24.79, 24.2, 17.53, 16.3, 16.3, 13.66, 12.26, 11.01, 8.94],
    #  [8.27, 4.74, 5.7, 5.24, 5.24, 4.13, 5.25, 3.77, 2.54]),
    
    # ("LINKX",
    #  [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
    #  [32.99, 23.91, 19.9, 18.6, 17.44, 15.08, 14.99, 11.67, 10.86],
    #  [9.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.83, 0.0]),
    
    # ("MixHopGCN",
    #  [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
    #  [65.87, 58.62, 17.65, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12],
    #  [14.64, 11.24, 29.31, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    
    # ("ChebGCN",
    #  [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
    #  [28.71, 24.71, 21.38, 21.01, 19.82, 16.75, 15.35, 10.26, 9.77],
    #  [4.29, 5.39, 5.21, 4.68, 5.16, 4.98, 3.71, 2.32, 2.48]),
    
    ("Proposed w.o. D1", # 22.30
     [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
     [85.53, 80.43, 65.46, 62.64,  60.1, 49.77, 41.64, 38.49, 34.89],
     [14.99, 12.34, 4.98, 20.27, 10.21, 12.52, 3.52, 12.7, 5.32]),
    
    ("Proposed w.o D2", # 26.28
     [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
     [97.42, 58.13, 54.2, 49.86, 49.43, 45.12, 38.35, 35.64, 32.47],
     [1.74, 11.5, 17.44, 7.44, 10.93, 5.11, 4.47, 4.88, 7.89]),

    ("EO-GNN", 
     [0.81, 0.7, 0.63, 0.49, 0.4, 0.29, 0.19, 0.17, 0.1],
     [90.78, 84.41, 66.53, 64.51, 62.3, 57.83, 52.14, 47.36, 45.83],  
     [7.34, 1.87, 14.99, 20.85, 10.83, 2.04, 13.77, 6.13, 3.3]),
]


LABEL_SIZE = 28
TICK_SIZE = 28
LEGEND_SIZE = 24
FIGSIZE = (10, 8)
DPI = 300


colors = plt.cm.get_cmap('tab10', len(raw_cora_mrr_data))

new_alpha = np.arange(0.1, 1.0, 0.1)
interpolated_data = defaultdict(dict)

for model, alpha, best_valid, variance in raw_cora_mrr_data:

    interpolated_data[model]["alpha"] =  new_alpha
    interpolated_data[model]["best_valid"] = best_valid
    interpolated_data[model]["variance"] = variance

fig, ax = plt.subplots(figsize=(10, 6))

dashed_models = {"ChebGCN", "LINKX", "GIN"}  # Models that will have dashed lines
line_styles = {model: "--" if model in dashed_models else "-" for model in interpolated_data.keys()}
alpha_values = {model: 0.5 if model in dashed_models else 1.0 for model in interpolated_data.keys()}  # Reduce opacity for dashed lines


model_colors["GAT"] = "pink"
model_colors["EO-GNN"] = "red"
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
new_alpha = np.arange(0.1, 1.0, 0.2)
ax.set_xlabel(r"$EAR$", fontsize=LABEL_SIZE)
ax.set_ylabel("MRR (/%)", fontsize=LABEL_SIZE)
ax.set_xticks(new_alpha)

ax.set_yticks(np.arange(0, 101, 20))
ax.set_ylim(30, 100) 
ax.tick_params(axis='both', labelsize=TICK_SIZE)

plt.tight_layout()
plt.savefig('ablation_Exp1_Cora_SYN_Real_MRR_updated.pdf')
print("Plot saved as 'ablation_Exp1_Cora_SYN_Real_MRR.pdf'")