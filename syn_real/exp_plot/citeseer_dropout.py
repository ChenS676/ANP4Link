
### citeseer no drop

# python real_syn_automorphic.py --data_name Citeseer --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 --runs 5 

# Citeseer_GCN_inter0.00_intra0.00_total0_Orbits_658.00_Norm_0.81_ArScore_0.10,17.83 ± 15.99,46.53 ± 11.91,*98.83 ± 0.14,*98.95 ± 0.08

# Citeseer_GCN_inter0.10_intra0.50_total200_Orbits_1157.00_Norm_0.80_ArScore_0.17,17.68 ± 13.19,33.46 ± 4.44,*98.09 ± 0.10,*98.39 ± 0.15

# Citeseer_GCN_inter0.10_intra0.50_total1000_Orbits_1840.00_Norm_0.76_ArScore_0.28,34.03 ± 16.21,44.56 ± 6.37,*95.34 ± 0.25,*96.78 ± 0.16

# Citeseer_GCN_inter0.10_intra0.50_total2000_Orbits_2522.00_Norm_0.73_ArScore_0.38,14.43 ± 10.16,44.80 ± 6.48,*92.62 ± 0.22,*94.97 ± 0.15

# Citeseer_GCN_inter0.10_intra0.50_total3000_Orbits_3255.00_Norm_0.70_ArScore_0.49,5.11 ± 4.34,21.66 ± 3.18,*89.85 ± 0.27,*92.96 ± 0.16

# Citeseer_GCN_inter0.10_intra0.50_total4000_Orbits_3822.00_Norm_0.65_ArScore_0.57,8.50 ± 2.97,19.48 ± 1.22,*88.50 ± 0.27,*91.56 ± 0.24

# Citeseer_GCN_inter0.10_intra0.50_total5000_Orbits_4377.00_Norm_0.62_ArScore_0.66,9.48 ± 11.13,23.94 ± 8.31,*86.22 ± 0.27,*90.42 ± 0.31

# Citeseer_GCN_inter0.10_intra0.50_total7000_Orbits_5121.00_Norm_0.53_ArScore_0.77,5.50 ± 4.55,19.18 ± 5.57,*82.25 ± 0.27,*87.09 ± 0.40

# Citeseer_GCN_inter0.10_intra0.50_total8000_Orbits_5350.00_Norm_0.50_ArScore_0.80,11.13 ± 5.14,21.95 ± 2.42,*80.97 ± 0.32,*86.00 ± 0.18

# Citeseer_GCN_inter0.10_intra0.50_total10000_Orbits_5778.00_Norm_0.43_ArScore_0.87,9.22 ± 4.96,15.38 ± 2.41,*77.59 ± 0.75,*82.75 ± 0.37

# Citeseer_GCN_inter0.10_intra0.50_total14000_Orbits_6264.00_Norm_0.29_ArScore_0.94,4.97 ± 2.90,9.84 ± 1.39,*72.45 ± 0.48,*78.14 ± 0.28

## dropedge

# python real_syn_automorphic.py --data_name Citeseer --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 --runs 5 --dropedge 0.1

# Citeseer_GCN_inter0.00_intra0.00_total0_Orbits_658.00_Norm_0.81_ArScore_0.10,16.73 ± 16.75,37.83 ± 9.05,*98.63 ± 0.21,*98.70 ± 0.18

# Citeseer_GCN_inter0.10_intra0.50_total200_Orbits_1157.00_Norm_0.80_ArScore_0.17,22.43 ± 15.17,38.80 ± 10.86,*98.16 ± 0.10,*98.48 ± 0.05

# Citeseer_GCN_inter0.10_intra0.50_total1000_Orbits_1840.00_Norm_0.76_ArScore_0.28,38.45 ± 10.22,57.55 ± 7.99,*95.33 ± 0.14,*96.71 ± 0.11

# Citeseer_GCN_inter0.10_intra0.50_total2000_Orbits_2522.00_Norm_0.73_ArScore_0.38,25.61 ± 6.31,37.79 ± 3.80,*92.76 ± 0.24,*94.96 ± 0.18

# Citeseer_GCN_inter0.10_intra0.50_total3000_Orbits_3255.00_Norm_0.70_ArScore_0.49,4.07 ± 2.76,22.10 ± 3.24,*89.83 ± 0.29,*92.82 ± 0.09

# Citeseer_GCN_inter0.10_intra0.50_total4000_Orbits_3822.00_Norm_0.65_ArScore_0.57,7.38 ± 4.70,18.93 ± 1.82,*88.09 ± 0.38,*91.64 ± 0.22

# Citeseer_GCN_inter0.10_intra0.50_total5000_Orbits_4377.00_Norm_0.62_ArScore_0.66,18.57 ± 12.62,26.79 ± 7.52,*86.63 ± 0.18,*90.47 ± 0.24

# Citeseer_GCN_inter0.10_intra0.50_total7000_Orbits_5121.00_Norm_0.53_ArScore_0.77,7.40 ± 4.08,19.79 ± 1.86,*82.50 ± 0.48,*87.36 ± 0.19

# Citeseer_GCN_inter0.10_intra0.50_total8000_Orbits_5350.00_Norm_0.50_ArScore_0.80,14.24 ± 5.22,21.36 ± 3.78,*81.45 ± 0.33,*86.27 ± 0.18

# Citeseer_GCN_inter0.10_intra0.50_total10000_Orbits_5778.00_Norm_0.43_ArScore_0.87,7.40 ± 2.43,14.25 ± 1.70,*78.08 ± 0.31,*83.19 ± 0.28

# Citeseer_GCN_inter0.10_intra0.50_total14000_Orbits_6264.00_Norm_0.29_ArScore_0.94,5.27 ± 1.10,9.07 ± 0.36,*72.86 ± 0.51,*78.49 ± 0.35

# ## dropnode

# python real_syn_automorphic.py --data_name Citeseer --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 --runs 5 --dropnode 0.1

# Citeseer_GCN_inter0.00_intra0.00_total0_Orbits_658.00_Norm_0.81_ArScore_0.10,21.79 ± 11.29,39.73 ± 7.88,*98.48 ± 0.17,*98.76 ± 0.17

# Citeseer_GCN_inter0.10_intra0.50_total200_Orbits_1157.00_Norm_0.80_ArScore_0.17,16.83 ± 7.99,37.17 ± 7.58,*98.06 ± 0.24,*98.32 ± 0.14

# Citeseer_GCN_inter0.10_intra0.50_total1000_Orbits_1840.00_Norm_0.76_ArScore_0.28,30.18 ± 8.31,45.16 ± 4.36,*95.15 ± 0.19,*96.65 ± 0.17

# Citeseer_GCN_inter0.10_intra0.50_total2000_Orbits_2522.00_Norm_0.73_ArScore_0.38,19.09 ± 11.32,37.22 ± 3.37,*92.67 ± 0.22,*94.90 ± 0.17

# Citeseer_GCN_inter0.10_intra0.50_total3000_Orbits_3255.00_Norm_0.70_ArScore_0.49,7.57 ± 4.77,23.98 ± 4.44,*89.78 ± 0.33,*92.97 ± 0.17

# Citeseer_GCN_inter0.10_intra0.50_total4000_Orbits_3822.00_Norm_0.65_ArScore_0.57,7.05 ± 3.02,20.60 ± 4.87,*88.24 ± 0.08,*91.52 ± 0.11

# Citeseer_GCN_inter0.10_intra0.50_total5000_Orbits_4377.00_Norm_0.62_ArScore_0.66,9.11 ± 5.94,23.89 ± 4.30,*86.61 ± 0.25,*90.39 ± 0.28

# Citeseer_GCN_inter0.10_intra0.50_total7000_Orbits_5121.00_Norm_0.53_ArScore_0.77,3.75 ± 2.78,21.22 ± 1.37,*82.56 ± 0.41,*87.09 ± 0.22

# Citeseer_GCN_inter0.10_intra0.50_total8000_Orbits_5350.00_Norm_0.50_ArScore_0.80,18.30 ± 5.38,25.11 ± 4.11,*80.98 ± 0.25,*86.11 ± 0.21

# Citeseer_GCN_inter0.10_intra0.50_total10000_Orbits_5778.00_Norm_0.43_ArScore_0.87,6.94 ± 3.55,13.96 ± 2.75,*77.93 ± 0.46,*83.10 ± 0.32

# Citeseer_GCN_inter0.10_intra0.50_total14000_Orbits_6264.00_Norm_0.29_ArScore_0.94,5.46 ± 3.12,8.67 ± 1.52,*72.71 ± 0.33,*78.27 ± 0.15

# ## droppath

# python real_syn_automorphic.py --data_name Citeseer --gnn_model GCN --lr 0.01 --dropout 0.3 --l2 1e-4 --num_layers 1 --num_layers_predictor 3 --hidden_channels 128 --epochs 9999 --kill_cnt 10 --eval_steps 5 --batch_size 1024 --runs 5 --droppath 0.1

# _GCN_inter0.00_intra0.00_total0_Orbits_658.00_Norm_0.81_ArScore_0.10,14.37 ± 17.93,34.46 ± 5.87,*98.42 ± 0.22,*98.43 ± 0.26

# Citeseer_GCN_inter0.10_intra0.50_total200_Orbits_1157.00_Norm_0.80_ArScore_0.17,19.81 ± 7.26,39.61 ± 2.12,*97.92 ± 0.13,*98.30 ± 0.12

# Citeseer_GCN_inter0.10_intra0.50_total1000_Orbits_1840.00_Norm_0.76_ArScore_0.28,41.02 ± 23.02,57.90 ± 10.48,*95.33 ± 0.26,*96.69 ± 0.23

# Citeseer_GCN_inter0.10_intra0.50_total2000_Orbits_2522.00_Norm_0.73_ArScore_0.38,24.56 ± 9.44,37.09 ± 3.72,*92.62 ± 0.14,*94.71 ± 0.15

# Citeseer_GCN_inter0.10_intra0.50_total3000_Orbits_3255.00_Norm_0.70_ArScore_0.49,5.81 ± 4.21,22.68 ± 3.17,*89.81 ± 0.64,*92.51 ± 0.29

# Citeseer_GCN_inter0.10_intra0.50_total4000_Orbits_3822.00_Norm_0.65_ArScore_0.57,8.16 ± 4.40,20.57 ± 4.35,*87.95 ± 0.23,*91.26 ± 0.24

# Citeseer_GCN_inter0.10_intra0.50_total5000_Orbits_4377.00_Norm_0.62_ArScore_0.66,14.86 ± 13.49,27.90 ± 7.89,*86.34 ± 0.49,*90.27 ± 0.33

# Citeseer_GCN_inter0.10_intra0.50_total7000_Orbits_5121.00_Norm_0.53_ArScore_0.77,5.26 ± 1.68,16.74 ± 1.17,*82.53 ± 0.58,*87.06 ± 0.29

# Citeseer_GCN_inter0.10_intra0.50_total8000_Orbits_5350.00_Norm_0.50_ArScore_0.80,14.56 ± 5.10,21.76 ± 3.18,*80.72 ± 0.52,*85.90 ± 0.19

# Citeseer_GCN_inter0.10_intra0.50_total10000_Orbits_5778.00_Norm_0.43_ArScore_0.87,5.32 ± 3.42,12.22 ± 2.45,*77.40 ± 0.38,*82.66 ± 0.32

# Citeseer_GCN_inter0.10_intra0.50_total14000_Orbits_6264.00_Norm_0.29_ArScore_0.94,5.37 ± 1.11,9.28 ± 2.30,*72.70 ± 0.44,*78.23 ± 0.42


# ============================================
# Citeseer GCN Results (No Drop / DropEdge / DropNode / DropPath)
# ============================================
import numpy as np
import matplotlib.pyplot as plt
from cora_drop_plot_huang import _sorted_series, metric_keys
from cora_drop_chen import model_colors
TITLE_SIZE = 26
LABEL_SIZE = 35
TICK_SIZE = 35
LEGEND_SIZE = 26
LEGEND_TITLE_SIZE = 24
ANNOTATION_SIZE = 24
FIGSIZE = (10, 8)
DPI = 300
LEGENG_SIZE = 15


citeseer_no_drop = {
    "mrr": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [12.92, 29.47, 16.82, 21.67, 26.30, 15.11, 14.66, 14.03, 12.11, 5.98, 4.88],
        [0.64, 2.42, 1.84, 3.27, 6.80, 1.52, 2.72, 1.12, 1.37, 1.32, 1.17],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.35, 97.86, 95.49, 93.27, 91.62, 88.77, 86.22, 85.10, 84.54, 81.14, 77.90],
        [0.40, 0.30, 0.26, 0.83, 0.75, 1.09, 1.57, 3.03, 2.17, 3.86, 2.62],
    )
}

citeseer_dropedge = {
    "mrr": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [13.61, 11.74, 9.56, 8.58, 4.13, 4.08, 3.37, 3.50, 3.34, 1.98, 1.49],
        [2.07, 3.04, 0.83, 1.30, 0.24, 0.58, 1.03, 0.86, 0.97, 0.46, 0.45],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [97.25, 97.39, 93.22, 93.12, 88.78, 85.86, 83.57, 79.52, 76.95, 76.21, 73.17],
        [0.45, 0.13, 1.62, 1.19, 1.72, 1.65, 1.42, 1.29, 1.56, 1.16, 1.83],
    )
}


citeseer_dropnode = {
    "mrr": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [10.52, 23.86, 14.71, 21.23, 20.01, 13.66, 15.30, 11.07, 10.70, 6.19, 4.41],
        [0.99, 2.04, 1.41, 2.79, 2.83, 1.16, 1.32, 2.44, 0.86, 1.99, 0.15],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [97.29, 97.27, 95.70, 93.72, 92.31, 91.18, 87.46, 86.31, 85.68, 81.99, 78.79],
        [0.58, 0.15, 0.55, 0.50, 0.41, 0.45, 1.69, 3.18, 3.38, 5.21, 4.58],
    )
}


citeseer_droppath = {
    "mrr": (
        [0.10,   0.17,   0.28,  0.38,  0.49,  0.57,  0.66,   0.77,  0.80,  0.87, 0.94],
        [23.86,  23.86,  21.23, 20.01, 14.71, 13.66, 15.30, 11.07,  10.52, 10.70, 6.19],
        [0.99, 2.04, 1.41, 2.79, 2.83, 1.16, 1.32, 2.44, 0.86, 1.99, 0.15],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [97.29, 97.27, 95.70, 93.72, 92.31, 91.18, 87.46, 86.31, 85.68, 81.99, 78.79],
        [0.58, 0.15, 0.55, 0.50, 0.41, 0.45, 1.69, 3.18, 3.38, 5.21, 4.58],
    )
}


#TODO replace result here 
citeseer_d1 = {
    "mrr": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [57.90, 39.61, 37.09, 34.46,  22.68, 20.57, 27.90, 16.74, 21.76, 12.22, 9.28],
        [5.87, 2.12, 10.48, 3.72, 3.17, 4.35, 7.89, 1.17, 3.18, 2.45, 2.30],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.42, 97.92, 95.33, 92.62, 89.81, 87.95, 86.34, 82.53, 80.72, 77.40, 72.70],
        [0.22, 0.13, 0.26, 0.14, 0.64, 0.23, 0.49, 0.58, 0.52, 0.38, 0.44],
    )
}

citeseer_adaedge = {
    "mrr": (
        [0.10,   0.17,  0.28,  0.38,  0.49,  0.57,  0.66,  0.77,  0.80,  0.87, 0.94],
        [29.85,  28.87, 23.70, 11.13, 10.47, 20.58, 13.46, 11.66, 10.90, 5.36, 5.80],
        [1.01, 1.33, 0.40, 2.04, 1.61, 8.65, 2.55, 1.27, 0.29, 0.76, 0.57],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.29, 97.55, 96.33, 93.65, 91.35, 89.86, 88.80, 85.67, 85.53, 81.03, 79.61],
        [0.40, 0.49, 0.21, 1.21, 0.33, 1.53, 0.69, 2.48, 0.14, 1.85, 0.94],
    ),
}


citeseer_EO_GNN = {
    "mrr": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [59.61, 59.46, 57.90, 37.09, 32.68, 30.57, 27.90, 16.74, 21.76, 12.22, 9.28],
        [5.87, 2.12, 10.48, 3.72, 3.17, 4.35, 7.89, 1.17, 3.18, 2.45, 2.30],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.42, 97.92, 95.33, 92.62, 89.81, 87.95, 86.34, 82.53, 80.72, 77.40, 72.70],
        [0.22, 0.13, 0.26, 0.14, 0.64, 0.23, 0.49, 0.58, 0.52, 0.38, 0.44],
    )
}




models_citeseer = {
    "GCN": citeseer_no_drop,
    "DropEdge": citeseer_dropedge,
    "DropNode": citeseer_dropnode,
    "DropPath": citeseer_droppath,
    "AdaEdge": citeseer_adaedge,
    "D1": citeseer_EO_GNN,
}

fixed_xticks = [0.1, 0.3, 0.5, 0.7, 0.9]

def plot_metric_all_models(models, metric_key, title=None, savepath=None):
    """Plot one metric (hits/mrr/auc/ap) for all models on the same axes."""
    if metric_key not in metric_keys:
        raise ValueError(f"Unknown metric_key: {metric_key}")
    yname, legend_metric = metric_keys[metric_key]

    plt.figure(figsize=(10, 6)) #plt.subplots(figsize=(10, 6))
    global_max, global_min = -np.inf, np.inf  

    for name, m in models.items():
        
        try:
            ar, mean, std = m[metric_key]
        except KeyError as e:
            raise KeyError(f'Model "{name}" missing metric "{metric_key}"') from e

        x, y, s = _sorted_series(ar, mean, std)
        y = np.asarray(y, dtype=float)
        s = np.asarray(s, dtype=float)
        col = model_colors[name]
        # plot mean line and std band
        print(name, col)
        plt.plot(x, y, marker="o", linewidth=2, label=name, color=col)
        plt.errorbar(
            x,
            y,
            s,
            fmt='o',  
            color=col,
            alpha=0.3,  
            capsize=6,
            elinewidth=2,
            capthick=2
        )
        global_min = min(global_min, np.nanmin(y - s))
        global_max = max(global_max, np.nanmax(y + s))

        # update global extrema
        if y.size:
            cur_min = float(np.nanmin(y - s))
            cur_max = float(np.nanmax(y + s))
            global_min = min(global_min, cur_min)
            global_max = max(global_max, cur_max)

    plt.xlabel(r"$EAR$", fontsize=LABEL_SIZE)
    plt.ylabel(f"{yname} (/%)", fontsize=LABEL_SIZE)
    plt.xticks(fixed_xticks, [f"{x:.1f}" for x in fixed_xticks])
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

    # handle missing data safely
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


# ---- Plot with your existing function ----
base = "/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot"

plot_metric_all_models(
    models_citeseer, "mrr",
    title="Citeseer – GCN: MRR vs ArScore (All Models)",
    savepath=f"{base}/citeseer_all_models_mrr2.pdf",
)
plot_metric_all_models(
    models_citeseer, "auc",
    title="Citeseer – GCN: AUC vs ArScore (All Models)",
    savepath=f"{base}/citeseer_all_models_auc2.pdf",
)
