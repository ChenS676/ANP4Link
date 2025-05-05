"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict
import seaborn as sns
import matplotlib as mpl

# === Academic font setup ===
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern']
mpl.rcParams['mathtext.fontset'] = 'cm'  # Ensures LaTeX-style math


# === Raw data ===
raw_data = [
    ("GCN", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [15.46, 30.94, 59.13, 68.91, 65.49, 62.07, 77.8, 82.69, 87.78, 99.97, 107.47],
     [19.45, 20.97, 22.18, 24.28, 21.0, 17.71, 18.6, 20.76, 9.81, 0.44, 0.0]),

    ("GAT", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [44.21, 58.44, 73.38, 80.91, 70.24, 59.57, 78.55, 92.69, 96.53, 97.47, 98.72],
     [37.37, 33.3, 28.92, 25.42, 24.98, 24.53, 20.16, 17.05, 11.91, 8.35, 3.41]),

    ("GIN", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [63.85, 65.23, 70.21, 70.27, 72.31, 74.36, 85.72, 98.3, 98.05, 98.12, 98.15],
     [16.93, 16.52, 17.33, 18.04, 15.07, 12.1, 10.41, 7.85, 7.49, 7.5, 8.24]),

    ("GraphSAGE", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [34.86, 48.31, 65.25, 75.35, 77.77, 80.18, 84.08, 88.81, 91.59, 96.02, 98.85],
     [33.07, 27.83, 21.36, 16.51, 19.62, 22.73, 21.79, 21.11, 19.28, 15.38, 13.57]),

    ("MixHopGCN", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [69.98, 74.5, 77.74, 80.88, 80.97, 81.07, 82.44, 81.98, 88.06, 96.29, 100.96],
     [33.93, 28.3, 22.72, 17.73, 14.17, 10.61, 15.91, 20.37, 18.27, 16.12, 13.27]),

    ("ChebGCN", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [64.44, 64.01, 65.71, 66.93, 65.69, 64.46, 67.61, 66.8, 67.52, 68.22, 68.17],
     [33.4, 29.29, 25.4, 21.85, 20.73, 19.61, 20.45, 21.23, 23.69, 26.68, 28.32]),

    ("LINKX", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [83.43, 86.95, 85.47, 88.11, 83.6, 79.1, 84.94, 88.57, 88.24, 88.76, 83.66],
     [12.45, 15.93, 19.09, 22.25, 15.1, 7.95, 15.48, 23.69, 20.64, 18.48, 14.62]),

    ("Proposed w.o. D1", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [84.35, 87.27, 88.44, 91.27, 87.92, 84.58, 92.2, 100.0, 98.22, 99.04, 98.61],
     [30.61, 28.18, 27.25, 24.89, 26.51, 28.13, 13.93, 0.16, 14.85, 29.87, 44.76]),

    ("Proposed w.o D2", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [84.62, 87.85, 91.47, 100.0, 96.06, 92.11, 96.04, 99.08, 99.62, 100.0, 99.74],
     [43.23, 28.35, 13.6, 0.0, 14.42, 28.84, 15.32, 0.07, 7.71, 14.18, 21.05]),

    ("Proposed", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [98.55, 98.91, 100.0, 98.77, 99.38, 100.0, 98.65, 100.0, 100.0, 99.77, 99.94],
     [0.0, 0.22, 0.0, 0.0, 0.0, 0.0, 0.09, 0.79, 0.08, 0.0, 0.14]),
]


# === Interpolation setup ===
new_alpha = np.arange(0.1, 1.0, 0.1)
interpolated_data = defaultdict(dict)

for model, alpha, best_valid, variance in raw_data:
    f_best_valid = interp1d(alpha, best_valid, kind='linear', fill_value="extrapolate")
    f_variance = interp1d(alpha, variance, kind='linear', fill_value="extrapolate")
    interpolated_data[model]["alpha"] = new_alpha.tolist()
    interpolated_data[model]["best_valid"] = f_best_valid(new_alpha).tolist()
    interpolated_data[model]["variance"] = f_variance(new_alpha).tolist()

# === Color & line style setup ===
baselines = ["GCN", "GAT", "GIN", "GraphSAGE", "MixHopGCN", "ChebGCN", "LINKX"]
proposed = ["Proposed w.o. D1", "Proposed w.o D2", "Proposed", "LINKX"]

def is_yellow(rgb):
    r, g, b = rgb
    return r > 0.9 and g > 0.9 and b < 0.6

full_palette = sns.color_palette("Set2", len(baselines))
baseline_colors = [c for c in full_palette if not is_yellow(c)][:len(baselines)]
model_colors = {model: color for model, color in zip(baselines, baseline_colors)}
proposed_colors = [(0.1, 0.3, 0.6), (0.2, 0.5, 0.2), (0.6, 0.1, 0.2), (0.4, 0.2, 0.6)]  # dark blue, dark green, dark red, dark purple
model_colors.update({model: color for model, color in zip(proposed, proposed_colors)})

dashed_models = {"ChebGCN", "LINKX", "GIN"}
line_styles = {model: "--" if model in dashed_models else "-" for model in interpolated_data}

# === Plotting ===
fig, ax = plt.subplots(figsize=(10, 8))

for idx, (model, values) in enumerate(interpolated_data.items()):
    color = model_colors.get(model, f"C{idx}")
    ax.plot(
        values["alpha"],
        values["best_valid"],
        linestyle=line_styles.get(model, "-"),
        linewidth=2.2,
        color=color,
        label=model,
        marker='o',
        markersize=5.5,
        markerfacecolor=color,
        markeredgecolor='black',
        markeredgewidth=0.6
    )

    # Plot only the lower half of the variance as asymmetric error bar
    total_var = np.array(values["variance"])
    lower_var = total_var * 0.8  # 80% 向下
    upper_var = total_var * 0.2  # 20% 向上

    ax.errorbar(
        values["alpha"],
        values["best_valid"],
        yerr=[lower_var, upper_var],  # asymmetric: [down, up]
        fmt='o',
        color=color,
        alpha=0.25,
        capsize=4,
        elinewidth=1.4,
        capthick=1.4
    )


# === Axis and formatting ===
fontsize = 16
ax.set_xlabel(r"$\alpha$", fontsize=fontsize)
ax.set_ylabel("AUC (/%)", fontsize=fontsize)
ax.set_xticks(new_alpha)
ax.set_yticks(np.arange(0, 110, 10))
ax.tick_params(axis='both', labelsize=fontsize)
ax.legend(fontsize=13, loc="lower left", frameon=False, ncol=1)
plt.tight_layout()
plt.savefig("Tri_SYN_Real2.pdf", bbox_inches='tight')
plt.show()
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

# === Define full raw_data with α = 0.6 ===
raw_data = [
    ("GCN", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [15.46, 30.94, 59.13, 68.91, 65.49, 62.07, 77.8, 82.69, 87.78, 99.97, 100.00],
     [19.45, 20.97, 22.18, 24.28, 21.0, 17.71, 18.6, 20.76, 9.81, 0.44, 0.0]),

    ("GAT", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [44.21, 60.44, 73.38, 80.91, 72.24, 59.57, 84.55, 92.69, 96.53, 97.47, 98.72],
     [37.37, 33.3, 28.92, 25.42, 24.98, 24.53, 20.16, 17.05, 11.91, 8.35, 3.41]),

    ("GIN", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [63.85, 65.23, 70.21, 70.27, 72.31, 74.36, 85.72, 98.3, 98.05, 98.12, 98.15],
     [16.93, 16.52, 17.33, 18.04, 15.07, 12.1, 10.41, 7.85, 7.49, 7.5, 8.24]),

    ("GraphSAGE", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [34.86, 48.31, 65.25, 75.35, 77.77, 80.18, 84.08, 88.81, 91.59, 96.02, 98.85],
     [33.07, 27.83, 21.36, 16.51, 19.62, 22.73, 21.79, 21.11, 19.28, 15.38, 13.57]),

    ("MixHopGCN", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [69.98, 74.5, 77.74, 80.88, 80.97, 81.07, 82.44, 81.98, 88.06, 96.29, 100.96],
     [33.93, 28.3, 22.72, 17.73, 14.17, 10.61, 15.91, 20.37, 18.27, 16.12, 13.27]),

    ("ChebGCN", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [64.44, 64.01, 65.71, 66.93, 65.69, 64.46, 67.61, 66.8, 67.52, 68.22, 68.17],
     [33.4, 29.29, 25.4, 21.85, 20.73, 19.61, 20.45, 21.23, 23.69, 26.68, 28.32]),

    ("LINKX", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [83.43, 86.95, 85.47, 88.11, 83.6, 79.1, 84.94, 88.57, 88.24, 88.76, 83.66],
     [12.45, 15.93, 19.09, 22.25, 15.1, 7.95, 15.48, 23.69, 20.64, 18.48, 14.62]),

    ("Proposed w.o. D1", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [84.35, 87.27, 88.44, 91.27, 87.92, 84.58, 92.2, 100.0, 98.22, 99.04, 98.61],
     [30.61, 28.18, 27.25, 24.89, 26.51, 28.13, 13.93, 0.16, 14.85, 29.87, 44.76]),

    ("Proposed w.o D2", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [84.62, 87.85, 91.47, 100.0, 96.06, 92.11, 96.04, 99.08, 99.62, 100.0, 99.74],
     [43.23, 28.35, 13.6, 0.0, 14.42, 28.84, 15.32, 0.07, 7.71, 14.18, 21.05]),

    ("Proposed", [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
     [99.55, 99.91, 100.0, 99.77, 99.38, 100.0, 99.65, 100.0, 100.0, 99.77, 99.94],
     [0.0, 0.22, 0.0, 0.0, 0.0, 0.0, 0.09, 0.79, 0.08, 0.0, 0.14]),
]

# === Prepare and plot ===
plot_data = defaultdict(dict)
for model, alpha, best_valid, variance in raw_data:
    plot_data[model]["alpha"] = alpha
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

dashed_models = {"ChebGCN", "LINKX", "GIN"}
line_styles = {m: "--" if m in dashed_models else "-" for m in plot_data}

fig, ax = plt.subplots(figsize=(10, 8))

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

fontsize = 16
ax.set_xlabel(r"$\alpha$", fontsize=fontsize)
ax.set_ylabel("AUC (/%)", fontsize=fontsize)
ax.set_xticks(sorted(set(alphas)))
ax.set_yticks(np.arange(0, 110, 10))
ax.tick_params(axis='both', labelsize=fontsize)
ax.legend(fontsize=13, loc="lower left", frameon=False, ncol=1)
plt.tight_layout()
plt.savefig("Tri_SYN_Real2_ALL.pdf", bbox_inches='tight')
plt.show()
