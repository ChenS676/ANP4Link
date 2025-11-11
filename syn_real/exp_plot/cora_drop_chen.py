import numpy as np
import matplotlib.pyplot as plt

# ---- Put your existing result dicts above this line ----
TITLE_SIZE = 26
LABEL_SIZE = 35
TICK_SIZE = 35
LEGEND_SIZE = 26
LEGEND_TITLE_SIZE = 24
ANNOTATION_SIZE = 24
FIGSIZE = (10, 8)
DPI = 300
LEGENG_SIZE = 15

# ===============================
# 1) Cora – GCN (no drop)
# ===============================

cora_no_drop = {
    "hits": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81],  # normalized ArScores (optional)
        [9.80, 20.29, 26.00, 10.72, 11.27, 1.98, 5.49, 8.68],
        [6.85, 6.32, 18.38, 6.01, 18.42, 0.65, 3.94, 5.37],
    ),
    "mrr": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81],
        [30.73, 44.45, 45.63, 40.12, 29.46, 12.89, 15.08, 20.60],
        [5.28, 2.38, 12.87, 3.60, 14.09, 0.99, 2.07, 1.98],
    ),
    "auc": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81],
        [99.66, 99.75, 99.61, 99.40, 98.86, 98.28, 97.88, 98.14],
        [0.05, 0.03, 0.08, 0.13, 0.08, 0.10, 0.28, 0.19],
    ),
    "ap": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81],
        [99.40, 99.66, 99.52, 99.40, 98.76, 97.99, 97.67, 98.11],
        [0.20, 0.03, 0.14, 0.11, 0.17, 0.12, 0.28, 0.23],
    ),
}


cora_adaedge = {
    "hits": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [12.82, 38.24, 7.09, 30.67, 22.64, 15.46, 0.69, 3.18, 5.44],
        [6.84, 4.86, 8.93, 2.03, 9.50, 9.75, 0.14, 1.43, 2.34],
    ),
    "mrr": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [32.48, 55.42, 33.07, 39.65, 39.40, 30.34, 17.94, 22.99, 18.94],
        [6.54, 1.58, 4.79, 2.64, 4.75, 11.19, 3.62, 1.72, 2.46],
    ),
    "auc": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [99.65, 99.80, 99.67, 99.21, 99.20, 98.85, 97.89, 97.81, 96.66],
        [0.03, 0.02, 0.02, 0.06, 0.02, 0.11, 0.06, 0.28, 0.15],
    ),
    "ap": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [99.45, 99.76, 99.35, 99.10, 99.20, 98.89, 97.82, 97.92, 96.89],
        [0.11, 0.02, 0.13, 0.05, 0.07, 0.09, 0.28, 0.24, 0.26],
    ),
}


# 说明：如需严格升序的 ArScore，请将 cora_dropedge 中的 ArScore 与对应数值
# 重新排序为 [0.19, 0.26, 0.37, 0.49, 0.60, 0.72, 0.80, 0.82, 0.88] 并同步调整均值/方差顺序。

# ===============================
# 3) Cora – GCN + DropNode(0.1)
# ===============================
cora_dropnode = {
    "hits": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [12.42, 22.95, 35.00, 25.24, 8.43, 11.35, 5.86, 4.79, 18.38],
        [3.68, 13.59, 17.37, 7.63, 6.93, 18.90, 1.09, 3.05, 7.04],
    ),
    "mrr": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [30.83, 43.70, 46.72, 38.03, 28.77, 26.65, 25.86, 14.49, 28.87],
        [3.97, 7.76, 15.45, 6.65, 10.20, 13.33, 0.72, 0.49, 4.42],
    ),
    "auc": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [99.64, 99.71, 99.66, 99.32, 99.25, 98.91, 98.20, 97.92, 97.60],
        [0.03, 0.02, 0.08, 0.05, 0.01, 0.04, 0.17, 0.29, 0.10],
    ),
    "ap": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [99.40, 99.65, 99.63, 99.32, 98.99, 98.72, 98.35, 97.76, 97.78],
        [0.11, 0.07, 0.06, 0.03, 0.33, 0.11, 0.05, 0.22, 0.11],
    ),
}

# ===============================
# 4) Cora – GCN + DropPath(0.1)
# ===============================
cora_droppath = {
    "hits": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],  # total_edges
        [9.14, 7.98, 2.81, 9.09, 7.85, 5.56, 8.11, 4.72, 3.01],  # Hits@1
        [5.90, 9.11, 2.43, 3.01, 6.57, 4.18, 0.41, 1.83, 1.49],  # std
    ),
    "mrr": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [17.16, 21.45, 15.44, 17.65, 16.27, 14.56, 13.74, 9.35, 7.57],
        [4.30, 4.38, 3.23, 1.60, 5.05, 1.13, 0.65, 1.21, 0.37],
    ),
    "auc": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [97.90, 98.39, 97.74, 96.86, 96.03, 94.54, 92.74, 91.99, 89.55],
        [0.31, 0.16, 0.09, 0.05, 0.54, 0.12, 0.09, 0.46, 0.66],
    ),
    "ap": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [97.26, 97.95, 97.10, 96.54, 95.74, 94.14, 92.43, 91.44, 88.75],
        [0.67, 0.22, 0.30, 0.19, 0.49, 0.27, 0.14, 0.37, 0.39],
    ),
}



# ===============================
# 2) Cora – GCN + DropEdge(0.1)
# ===============================

cora_dropedge = {
    "hits": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89], # total_edges
        [9.14, 7.98, 2.81, 9.09, 7.85, 5.56, 8.11, 4.72, 3.01],  # Hits@1
        [5.90, 9.11, 2.43, 3.01, 6.57, 4.18, 0.41, 1.83, 1.49],  # std
    ),
    "mrr": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [17.16, 21.45, 15.44, 17.65, 16.27, 14.56, 13.74, 9.35, 7.57],
        [4.30, 4.38, 3.23, 1.60, 5.05, 1.13, 0.65, 1.21, 0.37],
    ),
    "auc": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [97.90, 98.39, 97.74, 96.86, 96.03, 94.54, 92.74, 91.99, 89.55],
        [0.31, 0.16, 0.09, 0.05, 0.54, 0.12, 0.09, 0.46, 0.66],
    ),
    "ap": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [97.26, 97.95, 97.10, 96.54, 95.74, 94.14, 92.43, 91.44, 88.75],
        [0.67, 0.22, 0.30, 0.19, 0.49, 0.27, 0.14, 0.37, 0.39],
    ),
}


models = {
    "GCN":  cora_no_drop,
    "DropEdge": cora_dropedge,
    "DropNode": cora_dropnode,
    "DropPath": cora_droppath,
    "AdaEdge": cora_adaedge,
}

model_colors = {
    "GCN": (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),  # Dark orange
    "DropEdge": (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
    "DropNode": (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0),
    "DropPath":  (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0),
    "AdaEdge": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
}



metric_keys = {
    "hits": ("Hits@1", "Hits@1"),
    "mrr":  ("MRR",    "MRR"),
    "auc":  ("AUC",    "AUC"),
    "ap":   ("AP",     "AP"),
}

def _sorted_series(ar, mean, std):
    """Sort by ArScore ascending to make lines monotonic in x."""
    ar = np.array(ar, dtype=float)
    mean = np.array(mean, dtype=float)
    std = np.array(std, dtype=float)
    idx = np.argsort(ar)
    return ar[idx], mean[idx], std[idx]



def plot_metric_all_models(models, metric_key, title=None, savepath=None):
    """Plot one metric (hits/mrr/auc/ap) for all models on the same axes."""
    if metric_key not in metric_keys:
        raise ValueError(f"Unknown metric_key: {metric_key}")
    yname, legend_metric = metric_keys[metric_key]

    plt.figure(figsize=(7.5, 5.5))
    global_max, global_min = -np.inf, np.inf  # track global min & max

    for name, m in models.items():
        try:
            ar, mean, std = m[metric_key]
        except KeyError as e:
            raise KeyError(f'Model "{name}" missing metric "{metric_key}"') from e

        x, y, s = _sorted_series(ar, mean, std)
        y = np.asarray(y, dtype=float)
        s = np.asarray(s, dtype=float)

        # choose color
        col = model_colors[name]

        # plot mean line and std band with the chosen color
        plt.plot(x, y, marker="o", linewidth=2, label=name, color=col)
        plt.fill_between(x, y - s, y + s, alpha=0.15, linewidth=0, color=col)

        # update global extrema using mean ± std
        if y.size:
            cur_min = float(np.nanmin(y - s))
            cur_max = float(np.nanmax(y + s))
            global_min = min(global_min, cur_min)
            global_max = max(global_max, cur_max)

    plt.xlabel(r"$EAR$", fontsize=LABEL_SIZE)
    plt.ylabel(f"{yname} (/%)", fontsize=LABEL_SIZE)
    # plt.title(title if title else None)
    plt.grid(True, alpha=0.3)

    # ---- Adaptive legend position ----
    if metric_key in ["auc", "ap"]:
        legend_loc = "lower left"
        legend_anchor = (0.05, 0.05)
    else:
        legend_loc = "upper right"
        legend_anchor = None

    plt.legend(
        fontsize=LEGEND_SIZE,
        loc=legend_loc,
        bbox_to_anchor=legend_anchor,
        ncol=1,
        frameon=True,
        framealpha=0.5,
        fancybox=True,
    )

    plt.tick_params(axis='x', labelsize=TICK_SIZE)
    plt.tick_params(axis='y', labelsize=TICK_SIZE)

    # ---- Adaptive y-ticks: start from min, ensure ≥3 intervals, nice steps ----
    def _nice_step(target_range, min_intervals=3):
        if target_range <= 0:
            return 1.0
        raw = target_range / min_intervals
        exp = np.floor(np.log10(raw))
        frac = raw / (10 ** exp)
        if frac <= 1:   nice_frac = 1
        elif frac <= 2: nice_frac = 2
        elif frac <= 2.5: nice_frac = 2.5
        elif frac <= 5: nice_frac = 5
        else:           nice_frac = 10
        return nice_frac * (10 ** exp)

    if not np.isfinite(global_min):
        global_min = 0.0
    if not np.isfinite(global_max):
        global_max = 1.0

    y_range = max(0.0, global_max - global_min)
    step = _nice_step(y_range, min_intervals=3)
    ymin_rounded = np.floor(global_min / step) * step
    ymax_rounded = np.ceil(global_max / step) * step
    plt.ylim(ymin_rounded, ymax_rounded)
    plt.yticks(np.arange(ymin_rounded, ymax_rounded + 0.5 * step, step))

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=180)
    plt.show()


# ---- Generate the four figures ----
plot_metric_all_models(
    models, "hits",
    title="",
    savepath="/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot/chen_cora_all_models_hits2.png",
)
plot_metric_all_models(
    models, "mrr",
    title="",
    savepath="/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot/chen_cora_all_models_mrr2.png",
)
plot_metric_all_models(
    models, "auc",
    title="",
    savepath="/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot/chen_cora_all_models_auc2.png",
)
plot_metric_all_models(
    models, "ap",
    title="",
    savepath="/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot/chen_cora_all_models_ap2.png",
)
