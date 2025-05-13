import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict

raw_cora_auc_data = [
    ("GCN",
     [0.81, 0.70, 0.63, 0.49, 0.40, 0.29, 0.19, 0.17, 0.10],
     [99.41, 99.39, 99.06, 98.51, 98.01, 97.27, 96.45, 96.29, 94.07],
     [0.23, 0.09, 0.22, 0.35, 0.45, 0.42, 0.57, 0.67, 0.86]),

    # ("GAT",
    #  [0.81, 0.70, 0.63, 0.49, 0.40, 0.29, 0.19, 0.17, 0.10],
    #  [99.28, 98.97, 98.99, 98.75, 98.48, 97.98, 97.72, 97.51, 96.42],
    #  [0.16, 0.23, 0.13, 0.20, 0.25, 0.22, 0.50, 0.50, 0.39]),

     
    #arscore[0.8875553914327917, 0.8223781388478582, 0.797082717872969, 
    # 0.6959010339734121, 0.5939807976366323, 0.5088626292466765, 
    # 0.38404726735598227, 0.3059453471196455, 0.18593057607090102]

     # bs1024_lr0.001_testbs2048
    # ("BUDDY",
    #  [0.81,  0.69,  0.62,  0.49,  0.41, 0.30, 0.20, 0.18, 0.11],
    #  [95.34, 94.00, 95.03, 95.34, 94.82, 94.63, 94.06, 93.94, 93.80],
    #  [0.17, 0.23, 0.17, 0.04, 0.18, 0.30, 0.18, 0.10, 0.10]),
     
    # ("GIN",
    #  [0.81, 0.70, 0.63, 0.49, 0.40, 0.29, 0.19, 0.17, 0.10],
    #  [91.83, 95.76, 95.53, 94.91, 84.81, 82.49, 70.24, 72.01, 65.24],
    #  [0.70, 0.78, 0.59, 0.90, 0.14, 0.14, 0.78, 0.96, 0.38]),

    # ("GraphSAGE",
    #  [0.81, 0.70, 0.63, 0.49, 0.40, 0.29, 0.19, 0.17, 0.10],
    #  [97.91, 97.96, 97.96, 97.05, 95.93, 93.38, 90.27, 90.37, 85.82],
    #  [0.62, 0.57, 0.60, 0.85, 0.93, 0.99, 1.93, 2.24, 2.03]),

    # ("MixHopGCN",
    #  [0.81, 0.70, 0.63, 0.49, 0.40, 0.29, 0.19, 0.17, 0.10],
    #  [99.55, 99.28, 98.99, 98.75, 98.48, 97.98, 97.72, 97.51, 96.42],
    #  [0.17, 0.09, 0.13, 0.20, 0.25, 0.22, 0.50, 0.50, 0.39]),

    # ("ChebGCN",
    #  [0.81, 0.70, 0.63, 0.50, 0.49, 0.40, 0.29, 0.19, 0.17, 0.10],
    #  [98.41, 98.19, 97.82, 96.56, 96.76, 94.09, 91.18, 86.24, 84.88, 77.29],
    #  [0.13, 0.22, 0.21, 0.79, 0.32, 0.56, 1.02, 1.77, 1.42, 3.20]),

    # ("LINKX",
    #  [0.81,  0.69,  0.62,  0.49,  0.41, 0.30, 0.20, 0.18, 0.11],
    #  [97.30, 96.99, 96.81, 94.28, 92.82, 89.63, 86.69, 85.47, 83.48],
    #  [0.32, 0.15, 0.18, 0.15,  0.13,  0.29, 0.15, 0.12, 0.18]),

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
     [99.94, 99.86, 99.83, 99.81, 99.8, 99.78, 99.76, 99.74, 99.93],
     [0.03, 0.03, 0.07, 0.0, 0.05, 0.04, 0.03, 0.07, 0.08]),
]

TITLE_SIZE = 26
LABEL_SIZE = 35
TICK_SIZE = 35
LEGEND_SIZE = 26
LEGEND_TITLE_SIZE = 24
ANNOTATION_SIZE = 24
FIGSIZE = (10, 8)
DPI = 300


import seaborn as sns
model_list = ["GCN", "GAT", "GIN", "GraphSAGE", "MixHopGCN", "ChebGCN", "LINKX", "BUDDY", "Proposed", "Proposed w.o. D1", "Proposed w.o D2"]

new_alpha = np.arange(0.1, 1.0, 0.2)
interpolated_data = defaultdict(dict)
raw_data = raw_cora_auc_data
for model, alpha, best_valid, variance in raw_data:

    interpolated_data[model]["alpha"] =  [1-i for i in alpha] 
    interpolated_data[model]["best_valid"] = best_valid
    interpolated_data[model]["variance"] = variance

fig, ax = plt.subplots(figsize=(10, 6))

# Manually select color indices, **skipping yellow** (usually index 3 in 'tab10')
# tab10 index meanings:
# 0: blue, 1: orange, 2: green, 3: **yellow**, 4: red, 5: purple, 6: brown, 7: pink, 8: gray, 9: cyan
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

model_colors["GAT"] = "pink"
for idx, (model, values) in enumerate(interpolated_data.items()):
    color = model_colors.get(model, f"C{idx}")  # consistent color fallback

    # Plot main lines
    ax.plot(
        values["alpha"],
        values["best_valid"],
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
    

legend = ax.legend(
    fontsize=LEGEND_SIZE,
    loc="upper right",           # 位置改成右上角
    bbox_to_anchor=(0.95, 0.95), # 锚点靠近图的右上角，稍微留一点边
    ncol=1,
    frameon=True,
    framealpha=0.5,
    fancybox=True
)


legend.get_frame().set_facecolor('white')
ax.set_xlabel(r"$\alpha_{\mathcal{V}}$", fontsize=LABEL_SIZE)
ax.set_ylabel("AUC (/%)", fontsize=LABEL_SIZE)
ax.set_xticks(new_alpha)
ymin = 99.6
ymax = 100.02
yticks = np.linspace(ymin, ymax, 5)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{tick:.2f}" for tick in yticks])  # 保留两位小数

ax.set_ylim(ymin, 100.02) 
ax.tick_params(axis='both', labelsize=TICK_SIZE)
# ax.legend(fontsize=LEGENG_SIZE, loc="lower left")
plt.tight_layout()

plt.savefig('ablation_Exp1_Cora_SYN_AUC_Real.pdf')
