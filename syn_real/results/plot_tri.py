
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict

# Define sample data (replace this with your actual dataset)
raw_data = [
    ("GCN", [0.9, 0.7, 0.5, 0.3, 0.1], 
            [32.5, 67.5, 65.0, 82.5, 100.0], 
            [20.58, 23.72, 17.48, 20.58, 0.0]),
    ("GAT", [0.9, 0.7, 0.5, 0.3, 0.1], 
            [55.0, 77.5, 62.5, 92.5, 97.5], 
            [32.91, 24.86, 24.3, 16.87, 7.91]),
    ("GIN", [0.9, 0.7, 0.5, 0.3, 0.1], 
            [66.25, 70.5, 75.0, 97.5, 97.5], 
            [16.72, 18.45, 11.79, 7.91, 7.91]),
    ("GraphSAGE", [0.9, 0.7, 0.5, 0.3, 0.1], 
                [47.5, 75.0, 80.0, 87.5, 95.0], 
                [27.51, 16.67, 22.97, 21.25, 15.81]),
    ("MixHopGCN", [0.9, 0.7, 0.5, 0.3, 0.1], 
                [75.0, 80.0, 80.0, 82.5, 95.0], 
                [28.87, 17.48, 10.54, 20.58, 15.81]),
    ("ChebGCN", [0.9, 0.7, 0.5, 0.3, 0.1], 
                [65.0, 67.5, 66.25, 67.5, 67.5], 
                [29.34, 21.89, 20.45, 21.89, 26.48]),
    ("LINKX", [0.9, 0.7, 0.5, 0.3, 0.1], 
            [85.0, 87.0, 80.5, 90.0, 87.5], 
            [15.81, 22.97, 7.91, 23.57, 17.68]),
    
    ("Proposed w.o. D1", [0.9, 0.7, 0.5, 0.3, 0.1], 
     [87.0, 89.23,  83.33, 100.0, 100.0], 
     [28.87, 25.0, 28.87, 0.0, 28.87]),

    ("Proposed w.o D2", [0.9, 0.7, 0.5, 0.3, 0.1], 
     [89.33, 100.0, 93.33, 100.0, 99.67], 
     [28.87, 0.0, 28.87, 0.0, 14.43]),

    ("Proposed", [0.9, 0.7, 0.5, 0.3, 0.1], 
     [100.0, 100.0, 100.0, 100.0, 100.0], 
     [0.0, 0.0, 0.0, 0.0, 0.0]),

]
TITLE_SIZE = 26
LABEL_SIZE = 35
TICK_SIZE = 35
LEGEND_SIZE = 20
LEGEND_TITLE_SIZE = 18
ANNOTATION_SIZE = 24
FIGSIZE = (10, 8)
DPI = 300
LEGENG_SIZE = 15

# Define new interpolated alpha values
new_alpha = np.arange(0.1, 1.0, 0.2)

# Create a new dictionary to store interpolated results
interpolated_data = defaultdict(dict)

# Perform interpolation for each model
for model, alpha, best_valid, variance in raw_data:

    f_best_valid = interp1d(alpha, best_valid, kind='linear', fill_value="extrapolate")
    f_variance = interp1d(alpha, variance, kind='linear', fill_value="extrapolate")

    interpolated_data[model]["alpha"] = (new_alpha).tolist()
    interpolated_data[model]["best_valid"] = f_best_valid(new_alpha).tolist()
    interpolated_data[model]["variance"] = f_variance(new_alpha).tolist()

# Create the updated plot with error bars and reduced transparency for error bars
fig, ax = plt.subplots(figsize=(10, 8))

# Use 'tab10' colormap for distinguishable colors
colors = plt.cm.get_cmap('tab10', len(interpolated_data))

# Define different line styles and transparency settings
dashed_models = {"ChebGCN", "LINKX", "GIN"}  # Models that will have dashed lines
line_styles = {model: "--" if model in dashed_models else "-" for model in interpolated_data.keys()}
alpha_values = {model: 0.5 if model in dashed_models else 1.0 for model in interpolated_data.keys()}  # Reduce opacity for dashed lines

import seaborn as sns
baselines = ["GCN", "GAT", "GIN", "GraphSAGE", "MixHopGCN", "ChebGCN", "LINKX"]
proposed = ["Proposed w.o. D1", "Proposed w.o D2", "Proposed"]

def is_yellow(rgb):
    # Rough filter to exclude colors that are mostly yellowish (R ~ G >> B)
    r, g, b = rgb
    return r > 0.9 and g > 0.9 and b < 0.6

full_palette = sns.color_palette("Set2", len(baselines))
proposed_full_palette = sns.color_palette("pastel")

set3_colors = [color for color in full_palette if not is_yellow(color)]
set3_colors = set3_colors[:len(baselines)]  # Trim to match number of baselines
model_colors = {model: color for model, color in zip(baselines, set3_colors)}

proposed_colors = proposed_full_palette[:2]
proposed_colors = {model: color for model, color in zip(proposed, proposed_colors)}
model_colors.update(proposed_colors)
for idx, (model, values) in enumerate(interpolated_data.items()):

    color = model_colors.get(model, f"C{idx}")  # consistent color fallback
    # if model == 'GAT':
    #     color = 'blue'
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



ax.set_xlabel(r"$\alpha$", fontsize=LABEL_SIZE)
ax.set_ylabel("AUC (/%)", fontsize=LABEL_SIZE)
ax.set_xticks(new_alpha)
ax.set_yticks(np.arange(0, 102, 20))
ax.tick_params(axis='both', labelsize=LEGEND_SIZE) 
ax.legend(fontsize=LEGEND_SIZE , loc="lower left")
plt.tight_layout()

plt.savefig('Tri_SYN_Real.pdf')
