
import numpy as np
import matplotlib.pyplot as plt

# ---- Plot style constants ----
LABEL_SIZE = 28
TICK_SIZE = 28
LEGEND_SIZE = 24
FIGSIZE = (10, 8)
DPI = 300


# ===============================
# 1) Cora – GCN (no drop)
# ===============================
cora_no_drop = {
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
}

# ===============================
# 2) Cora – GCN + DropEdge(0.1) Chen 
# ===============================

cora_dropedge = {
    "mrr": (
        [0.19, 0.26, 0.37, 0.49, 0.60, 0.72, 0.80, 0.82, 0.88],
        [50.26, 39.08, 32.99, 28.11, 24.35, 16.28, 9.71, 18.11, 13.54],
        [11.31, 0.81, 3.69, 6.07, 4.20, 2.61, 0.61, 4.09, 3.41],
    ),
    "auc": (
        [0.19, 0.26, 0.37, 0.49, 0.60, 0.72, 0.80, 0.82, 0.88],
        [98.85, 98.54, 98.06, 97.87, 97.74, 97.76, 97.39, 97.07, 96.95],
        [0.10, 0.15, 0.14, 0.14, 0.22, 0.14, 0.27, 0.23, 0.39],
    ),
}

# ===============================
# 3) Cora – GCN + DropNode(0.1)
# ===============================
cora_dropnode = {
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
}

cora_adaedge = {
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
}


# ===============================
# 4) Cora – GCN + DropPath(0.1)
# ===============================

cora_droppath = {
    "mrr": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [21.99, 28.86, 18.76, 14.42, 18.19, 15.18, 17.59, 11.16, 14.54],
        [1.86, 5.02, 1.28, 1.03, 2.19, 1.87, 2.23, 0.90, 3.86],
    ),
    "auc": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [99.22, 99.24, 98.92, 98.80, 98.75, 98.69, 98.64, 98.59, 98.57],
        [0.26, 0.20, 0.24, 0.24, 0.26, 0.23, 0.20, 0.34, 0.36],
    ),
}



cora_drop_auto = {
    "mrr": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [62.48, 55.42, 48.07, 42.65, 39.40, 35.34, 30.94, 29.99, 28.94],
        [6.54, 1.58, 4.79, 2.64, 4.75, 11.19, 3.62, 1.72, 2.46],
    ),
    "auc": (
        [0.19, 0.27, 0.36, 0.50, 0.59, 0.70, 0.79, 0.81, 0.89],
        [99.85, 99.80, 99.77, 99.61, 99.50, 99.25, 99.10, 98.91, 98.66],
        [0.03, 0.02, 0.02, 0.06, 0.02, 0.11, 0.06, 0.28, 0.15],
    ),
}

# ---- Models and colors ----

models = {
    "GCN":  cora_no_drop,
    "DropEdge": cora_dropedge,
    "DropNode": cora_dropnode,
    "DropPath": cora_droppath,
    "AdaEdge": cora_adaedge,
    "D1": cora_drop_auto,
}


model_colors = {
    "GCN": (0.5, 0.5, 0.5, 1.0),
    "DropEdge": (1.0, 0.5, 0.05, 1.0),
    "DropNode": (0.74, 0.74, 0.13, 1.0),
    "DropPath": (0.09, 0.75, 0.81, 1.0),
    "AdaEdge": (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
    "D1": (0.35, 0.20, 0.75, 1.0)
}

metric_keys = {"mrr": ("MRR", "MRR"), "auc": ("AUC", "AUC")}


def _sorted_series(ar, mean, std):
    ar, mean, std = map(np.array, (ar, mean, std))
    idx = np.argsort(ar)
    return ar[idx], mean[idx], std[idx]


# ---- Fixed x-ticks for all plots ----
fixed_xticks = [0.1, 0.3, 0.5, 0.7, 0.9]


def plot_metric_all_models(models, metric_key, savepath=None):
    """Plot one metric with fixed x-ticks (0.1, 0.3, 0.5, 0.7, 0.9) and 4 equal y-intervals."""
    yname, _ = metric_keys[metric_key]
    plt.figure(figsize=(10, 6)) #plt.subplots(figsize=(10, 6))
    global_max, global_min = -np.inf, np.inf

    for name, m in models.items():
        if metric_key not in m:
            continue
        ar, mean, std = m[metric_key]
        x, y, s = _sorted_series(ar, mean, std)
        col = model_colors[name]
        plt.plot(x, y, marker="o", linewidth=2, label=name, color=col)
        # plt.fill_between(x, y - s, y + s, alpha=0.15, linewidth=0, color=col)
        plt.errorbar(
            x,
            y,
            s,
            fmt='o',  # Markers only for error bars
            color=col,
            alpha=0.3,  # Reduced transparency for error bars
            capsize=6,
            elinewidth=2,
            capthick=2
        )
        global_min = min(global_min, np.nanmin(y - s))
        global_max = max(global_max, np.nanmax(y + s))

    plt.xlabel(r"$EAR$", fontsize=LABEL_SIZE)
    plt.ylabel(f"{yname} (/%)", fontsize=LABEL_SIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(
    fontsize=LEGEND_SIZE,
    loc="upper right",
    ncol=2,
    frameon=True,
    framealpha=0.5,
    fancybox=True
    )

    plt.tick_params(axis='x', labelsize=TICK_SIZE)
    plt.tick_params(axis='y', labelsize=TICK_SIZE)

    # ---- Apply fixed x-ticks ----
    plt.xticks(fixed_xticks, [f"{x:.1f}" for x in fixed_xticks])
    plt.xlim(0.1, 0.9)

    # ---- Compute y-ticks with exactly 4 equal intervals ----
    y_min, y_max = global_min, global_max
    if y_min == y_max:
        y_min -= 1
        y_max += 1
    y_ticks = np.linspace(y_min, y_max, 5)
    plt.ylim(y_min, y_max)
    plt.yticks(y_ticks, [f"{t:.2f}" for t in y_ticks])

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=DPI)
    plt.show()


# ---- Generate all four figures ----
base = "/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot"
plot_metric_all_models(models, "mrr", savepath=f"{base}/cora_all_models_mrr2.pdf")
plot_metric_all_models(models, "auc", savepath=f"{base}/cora_all_models_auc2.pdf")
