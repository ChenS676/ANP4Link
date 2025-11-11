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
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [0.23, 10.11, 5.28, 1.14, 5.04, 3.80, 8.84, 2.84, 7.57],
        [0.47, 4.09, 3.34, 0.89, 3.60, 1.25, 2.61, 1.40, 4.71],
    ),
    "mrr": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [19.69, 31.50, 20.54, 15.42, 15.28, 16.07, 17.69, 12.14, 14.36],
        [4.11, 5.23, 3.93, 4.01, 2.64, 1.65, 2.44, 1.40, 4.13],
    ),
    "auc": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [98.48, 98.45, 97.64, 95.10, 92.75, 90.12, 87.53, 86.15, 82.82],
        [0.21, 0.07, 0.23, 0.20, 0.34, 0.31, 0.29, 0.24, 0.48],
    ),
    "ap": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [98.17, 98.43, 97.71, 95.98, 94.27, 92.58, 90.58, 89.62, 86.61],
        [0.16, 0.18, 0.14, 0.07, 0.14, 0.19, 0.23, 0.16, 0.21],
    ),
}

# ===============================
# 2) Cora – GCN + DropEdge(0.1)
# ===============================
cora_dropedge = {
    "hits": (
        [0.19, 0.26, 0.37, 0.49, 0.60, 0.72, 0.80, 0.82, 0.88],
        [37.47, 7.96, 10.67, 16.08, 10.57, 3.85, 2.02, 8.26, 1.99],
        [18.68, 4.49, 5.12, 11.16, 6.46, 3.93, 0.96, 3.84, 1.46],
    ),
    "mrr": (
        [0.19, 0.26, 0.37, 0.49, 0.60, 0.72, 0.80, 0.82, 0.88],
        [50.26, 39.08, 32.99, 28.11, 24.35, 16.28, 9.71, 18.11, 13.54],
        [11.31, 0.81, 3.69, 6.07, 4.20, 2.61, 0.61, 4.09, 3.41],
    ),
    "auc": (
        [0.19, 0.26, 0.37, 0.49, 0.60, 0.72, 0.80, 0.82, 0.88],
        [98.85, 98.54, 98.06, 95.87, 94.04, 91.76, 87.39, 87.07, 82.95],
        [0.10, 0.15, 0.14, 0.14, 0.22, 0.14, 0.27, 0.23, 0.39],
    ),
    "ap": (
        [0.19, 0.26, 0.37, 0.49, 0.60, 0.72, 0.80, 0.82, 0.88],
        [98.96, 98.80, 98.28, 96.91, 95.38, 93.70, 90.07, 90.35, 87.19],
        [0.19, 0.05, 0.13, 0.07, 0.05, 0.07, 0.16, 0.22, 0.19],
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
        [0.23, 7.06, 5.95, 1.80, 7.08, 3.91, 12.44, 3.63, 7.19],
        [0.34, 5.68, 4.28, 2.56, 4.62, 2.03, 3.09, 2.02, 3.05],
    ),
    "mrr": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [22.10, 30.48, 24.23, 15.86, 17.79, 14.75, 20.71, 12.05, 16.24],
        [4.43, 8.64, 6.58, 3.29, 3.89, 2.29, 1.91, 2.74, 2.65],
    ),
    "auc": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [98.56, 98.41, 97.86, 95.19, 92.82, 90.19, 87.82, 86.44, 82.71],
        [0.11, 0.09, 0.22, 0.27, 0.03, 0.43, 0.32, 0.26, 0.29],
    ),
    "ap": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [98.32, 98.56, 97.97, 96.12, 94.41, 92.78, 90.90, 89.72, 87.03],
        [0.21, 0.12, 0.11, 0.18, 0.09, 0.24, 0.27, 0.20, 0.33],
    ),
}

# ===============================
# 4) Cora – GCN + DropPath(0.1)
# ===============================
cora_droppath = {
    "hits": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [0.15, 14.23, 6.33, 0.60, 6.15, 3.52, 8.42, 4.71, 4.35],
        [0.18, 14.01, 5.31, 0.65, 3.41, 0.65, 3.63, 3.84, 1.97],
    ),
    "mrr": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [21.99, 28.86, 18.76, 14.42, 18.19, 15.18, 17.59, 11.16, 14.54],
        [1.86, 5.02, 1.28, 1.03, 2.19, 1.87, 2.23, 0.90, 3.86],
    ),
    "auc": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [98.22, 98.24, 97.42, 94.90, 92.85, 89.99, 87.24, 85.89, 82.57],
        [0.26, 0.20, 0.24, 0.24, 0.26, 0.23, 0.20, 0.34, 0.36],
    ),
    "ap": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [98.22, 98.39, 97.53, 95.88, 94.46, 92.52, 90.57, 89.39, 86.40],
        [0.23, 0.28, 0.21, 0.12, 0.16, 0.18, 0.19, 0.13, 0.23],
    ),
}

model_colors = {
    "GCN": (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),  # Dark orange
    "DropEdge": (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
    "DropNode": (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0),
    "DropPath":  (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0),
    "D1": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0)
}

models = {
    "No Drop":  cora_no_drop,
    "DropEdge": cora_dropedge,
    "DropNode": cora_dropnode,
    "DropPath": cora_droppath,
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

# Map model display names to color keys in model_colors
def _color_key_for(name: str) -> str:
    n = name.strip().lower()
    if "no drop" in n or n == "gcn":
        return "GCN"
    if "dropedge" in n:
        return "DropEdge"
    if "dropnode" in n:
        return "DropNode"
    if "droppath" in n:
        return "DropPath"
    return name  # fallback: try exact name as key

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
        ck = _color_key_for(name)
        col = model_colors.get(ck, None)  # None -> matplotlib default if not found

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
    savepath="/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot/cora_all_models_hits2.png",
)
plot_metric_all_models(
    models, "mrr",
    title="",
    savepath="/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot/cora_all_models_mrr2.png",
)
plot_metric_all_models(
    models, "auc",
    title="",
    savepath="/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot/cora_all_models_auc2.png",
)
plot_metric_all_models(
    models, "ap",
    title="",
    savepath="/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot/cora_all_models_ap2.png",
)
