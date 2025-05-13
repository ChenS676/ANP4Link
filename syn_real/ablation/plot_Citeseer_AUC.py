import seaborn as sns
models = ["GCN", "GAT", "GIN", "GraphSAGE", "MixHopGCN", "ChebGCN", "LINKX"]

model_colors = {
    "GCN": (0.8509803921568627, 0.37254901960784315, 0.00784313725490196, 1.0),  # Dark orange (distinct)
    "GAT": (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
    "GIN": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
    "GraphSAGE": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0),
    "BUDDY": (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0),
    "MixHopGCN": (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
    "ChebGCN": (0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0),
    "LINKX": (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
    "Proposed w.o. D1": (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0),
    "Proposed w.o D2": (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0),
    "Proposed": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0)
}

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict

TITLE_SIZE = 26
LABEL_SIZE = 35
TICK_SIZE = 35
LEGEND_SIZE = 26
LEGEND_TITLE_SIZE = 24
ANNOTATION_SIZE = 24
FIGSIZE = (10, 8)
DPI = 300
LEGENG_SIZE = 15

# Define sample data (replace this with your actual dataset)
raw_citeseer_auc_data = [
    ("GCN",
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
     [99.65, 99.50, 99.42, 98.92, 98.61, 98.03, 97.65, 96.32, 96.49, 95.03, 91.81],
     [0.08, 0.14, 0.16, 0.24, 0.33, 0.22, 0.47, 0.56, 0.74, 1.60, 2.05]),

    # ("GAT",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
    #  [99.42, 98.94, 99.17, 98.28, 98.31, 97.89, 97.81, 96.56, 95.87, 93.11, 91.93],
    #  [0.14, 0.32, 0.14, 0.71, 0.74, 1.07, 0.41, 1.26, 1.41, 4.54, 2.09]),

    # ar_scores = [
    #     0.9358280733393447, 0.8755635707844905, 0.8128944995491434,
    #     0.7753231139164413, 0.6549443943492635, 0.5766456266907124,
    #     0.47971145175834085, 0.3826269912834385, 0.2777276825969342,
    #     0.18079350766456267, 0.09888788698527201
    # ]
     #bs1024_lr0.02_testbs2048
    # ("BUDDY",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
    #  [95.2, 95.83, 95.84, 94.37, 95.1, 95.79, 96.03, 94.86, 94.7, 95.5, 95.57],
    #  [0.39, 0.18, 0.26, 0.53, 0.51, 0.48, 0.32, 0.42, 0.31, 0.39, 0.41]),

    # ("GIN",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
    #  [95.43, 95.40, 94.69, 92.73, 90.53, 89.78, 86.46, 78.42, 82.29, 75.34, 61.25],
    #  [1.10, 0.71, 1.14, 0.79, 1.34, 1.16, 2.00, 1.10, 1.19, 1.54, 1.04]),

    # ("GraphSAGE",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
    #  [96.19, 95.99, 94.18, 93.26, 91.69, 91.81, 86.72, 85.62, 82.41, 77.59, 74.68],
    #  [2.00, 1.77, 1.73, 3.49, 2.63, 2.91, 4.49, 5.20, 6.02, 3.97, 5.96]),


    # ("MixHopGCN",
    # [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
    # [99.69, 99.41, 99.03, 98.77, 98.22, 97.99, 96.282, 93.72, 94.55, 91.21, 88.347],
    # [0.06, 1.60, 0.25, 0.07, 0.27, 0.23, 0.442, 0.76, 0.52, 1.95, 3.176]),


    # ("ChebGCN",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
    #  [97.59, 97.76, 97.20, 94.90, 92.50, 88.75, 84.72, 79.21, 76.92, 72.16, 68.81],
    #  [0.46, 0.41, 0.44, 0.58, 1.31, 1.86, 1.89, 1.42, 1.49, 2.07, 2.76]),

    # ("LINKX",
    # [0.9,   0.82,  0.72,  0.62,  0.52,  0.43,  0.35,  0.23,   0.19, 0.12, 0.06],
    # [97.30, 96.99, 96.81, 94.28, 92.82, 89.63, 86.69, 85.47, 83.48, 81.28, 79.68],
    # [0.08,  0.14,  0.16,  0.24,  0.33,  0.22,  0.47,  0.56,  0.74, 1.42, 1.49]),

    ('Proposed w.o. D1',
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
     [99.89, 99.83, 99.77, 99.76, 99.76, 99.74, 99.67, 99.63, 99.625, 99.6225, 99.62],
     [0.08, 0.1, 0.11, 0.07, 0.05, 0.07, 0.12, 0.07, 0.145, 0.1825, 0.22]),

    ('Proposed w.o D2',
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
     [99.9, 99.8, 99.79, 99.78, 99.75, 99.74, 99.73, 99.71, 99.7775, 99.83875, 99.73],
     [0.05, 0.05, 0.07, 0.12, 0.03, 0.07, 0.04, 0.03, 0.04, 0.045, 0.05]),

    ('Proposed',
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
     [99.94, 99.86, 99.83, 99.81, 99.8, 99.78, 99.76, 99.74, 99.735, 99.7325, 99.9],
     [0.03, 0.03, 0.07, 0.0, 0.05, 0.04, 0.03, 0.07, 0.075, 0.0775, 0.08])
    
]


# Define new interpolated alpha values
new_alpha = np.arange(0.1, 1.0, 0.2)

# Create a new dictionary to store interpolated results
interpolated_data = defaultdict(dict)

# Perform interpolation for each model
for model, alpha, best_valid, variance in raw_citeseer_auc_data:

    # f_best_valid = interp1d(alpha, best_valid, kind='linear', fill_value="extrapolate")
    # f_variance = interp1d(alpha, variance, kind='linear', fill_value="extrapolate")

    interpolated_data[model]["alpha"] =  [1-i for i in alpha] 
    interpolated_data[model]["best_valid"] = best_valid
    interpolated_data[model]["variance"] = variance

# Create the updated plot with error bars and reduced transparency for error bars
fig, ax = plt.subplots(figsize=(10, 6))

# Use 'tab10' colormap for distinguishable colors
colors = plt.cm.get_cmap('tab10', len(interpolated_data))

# Define different line styles and transparency settings
dashed_models = {"ChebGCN", "LINKX", "GIN"}  # Models that will have dashed lines
line_styles = {model: "--" if model in dashed_models else "-" for model in interpolated_data.keys()}
alpha_values = {model: 0.5 if model in dashed_models else 1.0 for model in interpolated_data.keys()}  # Reduce opacity for dashed lines

model_colors["GAT"] = "pink"
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
        fmt='o',  # Markers only for error bars
        color=color,
        alpha=0.3,  # Reduced transparency for error bars
        capsize=6,
        elinewidth=2,
        capthick=2
    )
fontsize = 22
# Formatting the plot
# ax.set_xlabel(r"$\alpha_{\mathcal{E}}$", fontsize=LABEL_SIZE)
# ax.set_ylabel("AUC (/%)", fontsize=LABEL_SIZE)
# ax.set_xticks(new_alpha)
# ax.set_yticks(np.arange(99.5, 100, 0.25))
# ax.tick_params(axis='both', labelsize=TICK_SIZE)
# ax.legend(fontsize=LEGENG_SIZE, loc="lower left")
# plt.tight_layout()
# Formatting the plot
legend = ax.legend(
    fontsize=LEGEND_SIZE,
    loc="upper right",            # <<< 改成 "upper right"
    bbox_to_anchor=(0.95, 0.95),   # <<< (0,0) 左下角，(1,1) 右上角，这里留一点边
    ncol=1,                        # <<< 改成一列
    frameon=True,
    framealpha=0.5,
    fancybox=True
)

legend.get_frame().set_facecolor('white')
ax.set_xlabel(r"$\alpha_{\mathcal{V}}$", fontsize=LABEL_SIZE)
ax.set_ylabel("AUC (/%)", fontsize=LABEL_SIZE)
ax.set_xticks(new_alpha)
ymin = 99.6
ymax = 100
yticks = np.linspace(ymin, ymax, 5)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{tick:.2f}" for tick in yticks])  # 保留两位小数

ax.set_ylim(ymin, ymax)  # 🔥🔥🔥 这里设置只显示 99.62-100
ax.tick_params(axis='both', labelsize=TICK_SIZE)
# ax.legend(fontsize=LEGENG_SIZE, loc="lower left")
plt.tight_layout()

plt.savefig('ablation_Exp1_Citeseer_SYN_AUC_Real.pdf')