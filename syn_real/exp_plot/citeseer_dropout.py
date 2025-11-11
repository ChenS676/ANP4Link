
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
from syn_real.exp_plot.cora_drop_plot_huang import _sorted_series, metric_keys

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
    "hits": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [17.83, 17.68, 34.03, 14.43, 5.11, 8.50, 9.48, 5.50, 11.13, 9.22, 4.97],
        [15.99, 13.19, 16.21, 10.16, 4.34, 2.97, 11.13, 4.55, 5.14, 4.96, 2.90],
    ),
    "mrr": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [46.53, 33.46, 44.56, 44.80, 21.66, 19.48, 23.94, 19.18, 21.95, 15.38, 9.84],
        [11.91, 4.44, 6.37, 6.48, 3.18, 1.22, 8.31, 5.57, 2.42, 2.41, 1.39],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.83, 98.09, 95.34, 92.62, 89.85, 88.50, 86.22, 82.25, 80.97, 77.59, 72.45],
        [0.14, 0.10, 0.25, 0.22, 0.27, 0.27, 0.27, 0.27, 0.32, 0.75, 0.48],
    ),
    "ap": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.95, 98.39, 96.78, 94.97, 92.96, 91.56, 90.42, 87.09, 86.00, 82.75, 78.14],
        [0.08, 0.15, 0.16, 0.15, 0.16, 0.24, 0.31, 0.40, 0.18, 0.37, 0.28],
    ),
}

citeseer_dropedge = {
    "hits": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [16.73, 22.43, 38.45, 25.61, 4.07, 7.38, 18.57, 7.40, 14.24, 7.40, 5.27],
        [16.75, 15.17, 10.22, 6.31, 2.76, 4.70, 12.62, 4.08, 5.22, 2.43, 1.10],
    ),
    "mrr": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [37.83, 38.80, 57.55, 37.79, 22.10, 18.93, 26.79, 19.79, 21.36, 14.25, 9.07],
        [9.05, 10.86, 7.99, 3.80, 3.24, 1.82, 7.52, 1.86, 3.78, 1.70, 0.36],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.63, 98.16, 95.33, 92.76, 89.83, 88.09, 86.63, 82.50, 81.45, 78.08, 72.86],
        [0.21, 0.10, 0.14, 0.24, 0.29, 0.38, 0.18, 0.48, 0.33, 0.31, 0.51],
    ),
    "ap": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.70, 98.48, 96.71, 94.96, 92.82, 91.64, 90.47, 87.36, 86.27, 83.19, 78.49],
        [0.18, 0.05, 0.11, 0.18, 0.09, 0.22, 0.24, 0.19, 0.18, 0.28, 0.35],
    ),
}

citeseer_dropnode = {
    "hits": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [21.79, 16.83, 30.18, 19.09, 7.57, 7.05, 9.11, 3.75, 18.30, 6.94, 5.46],
        [11.29, 7.99, 8.31, 11.32, 4.77, 3.02, 5.94, 2.78, 5.38, 3.55, 3.12],
    ),
    "mrr": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [39.73, 37.17, 45.16, 37.22, 23.98, 20.60, 23.89, 21.22, 25.11, 13.96, 8.67],
        [7.88, 7.58, 4.36, 3.37, 4.44, 4.87, 4.30, 1.37, 4.11, 2.75, 1.52],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.48, 98.06, 95.15, 92.67, 89.78, 88.24, 86.61, 82.56, 80.98, 77.93, 72.71],
        [0.17, 0.24, 0.19, 0.22, 0.33, 0.08, 0.25, 0.41, 0.25, 0.46, 0.33],
    ),
    "ap": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.76, 98.32, 96.65, 94.90, 92.97, 91.52, 90.39, 87.09, 86.11, 83.10, 78.27],
        [0.17, 0.14, 0.17, 0.17, 0.17, 0.11, 0.28, 0.22, 0.21, 0.32, 0.15],
    ),
}

citeseer_droppath = {
    "hits": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [14.37, 19.81, 41.02, 24.56, 5.81, 8.16, 14.86, 5.26, 14.56, 5.32, 5.37],
        [17.93, 7.26, 23.02, 9.44, 4.21, 4.40, 13.49, 1.68, 5.10, 3.42, 1.11],
    ),
    "mrr": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [34.46, 39.61, 57.90, 37.09, 22.68, 20.57, 27.90, 16.74, 21.76, 12.22, 9.28],
        [5.87, 2.12, 10.48, 3.72, 3.17, 4.35, 7.89, 1.17, 3.18, 2.45, 2.30],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.42, 97.92, 95.33, 92.62, 89.81, 87.95, 86.34, 82.53, 80.72, 77.40, 72.70],
        [0.22, 0.13, 0.26, 0.14, 0.64, 0.23, 0.49, 0.58, 0.52, 0.38, 0.44],
    ),
    "ap": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.43, 98.30, 96.69, 94.71, 92.51, 91.26, 90.27, 87.06, 85.90, 82.66, 78.23],
        [0.26, 0.12, 0.23, 0.15, 0.29, 0.24, 0.33, 0.29, 0.19, 0.32, 0.42],
    ),
}

#TODO replace result here 
citeseer_d1 = {
    "hits": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [14.37, 19.81, 41.02, 24.56, 5.81, 8.16, 14.86, 5.26, 14.56, 5.32, 5.37],
        [17.93, 7.26, 23.02, 9.44, 4.21, 4.40, 13.49, 1.68, 5.10, 3.42, 1.11],
    ),
    "mrr": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [34.46, 39.61, 57.90, 37.09, 22.68, 20.57, 27.90, 16.74, 21.76, 12.22, 9.28],
        [5.87, 2.12, 10.48, 3.72, 3.17, 4.35, 7.89, 1.17, 3.18, 2.45, 2.30],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.42, 97.92, 95.33, 92.62, 89.81, 87.95, 86.34, 82.53, 80.72, 77.40, 72.70],
        [0.22, 0.13, 0.26, 0.14, 0.64, 0.23, 0.49, 0.58, 0.52, 0.38, 0.44],
    ),
    "ap": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.43, 98.30, 96.69, 94.71, 92.51, 91.26, 90.27, 87.06, 85.90, 82.66, 78.23],
        [0.26, 0.12, 0.23, 0.15, 0.29, 0.24, 0.33, 0.29, 0.19, 0.32, 0.42],
    ),
}

citeseer_EO_GNN = {
    "hits": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [14.37, 19.81, 41.02, 24.56, 5.81, 8.16, 14.86, 5.26, 14.56, 5.32, 5.37],
        [17.93, 7.26, 23.02, 9.44, 4.21, 4.40, 13.49, 1.68, 5.10, 3.42, 1.11],
    ),
    "mrr": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [34.46, 39.61, 57.90, 37.09, 22.68, 20.57, 27.90, 16.74, 21.76, 12.22, 9.28],
        [5.87, 2.12, 10.48, 3.72, 3.17, 4.35, 7.89, 1.17, 3.18, 2.45, 2.30],
    ),
    "auc": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.42, 97.92, 95.33, 92.62, 89.81, 87.95, 86.34, 82.53, 80.72, 77.40, 72.70],
        [0.22, 0.13, 0.26, 0.14, 0.64, 0.23, 0.49, 0.58, 0.52, 0.38, 0.44],
    ),
    "ap": (
        [0.10, 0.17, 0.28, 0.38, 0.49, 0.57, 0.66, 0.77, 0.80, 0.87, 0.94],
        [98.43, 98.30, 96.69, 94.71, 92.51, 91.26, 90.27, 87.06, 85.90, 82.66, 78.23],
        [0.26, 0.12, 0.23, 0.15, 0.29, 0.24, 0.33, 0.29, 0.19, 0.32, 0.42],
    ),
}



models_citeseer = {
    "No Drop":      citeseer_no_drop,
    "DropEdge": citeseer_dropedge,
    "DropNode": citeseer_dropnode,
    "DropPath": citeseer_droppath,
}


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

        # plot mean line and std band
        plt.plot(x, y, marker="o", linewidth=2, label=name)
        plt.fill_between(x, y - s, y + s, alpha=0.15, linewidth=0)

        # update global extrema
        if y.size:
            cur_min = float(np.nanmin(y - s))
            cur_max = float(np.nanmax(y + s))
            global_min = min(global_min, cur_min)
            global_max = max(global_max, cur_max)

    plt.xlabel(r"$EAR$", fontsize=LABEL_SIZE)
    plt.ylabel(f"{yname} (/%)", fontsize=LABEL_SIZE)
    plt.title(title if title else f"Cora – GCN: {legend_metric} vs ArScore")
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
plot_metric_all_models(
    models_citeseer, "hits",
    title="Citeseer – GCN: Hits@1 vs ArScore (All Models)",
    savepath="/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot/citeseer_all_models_hits2ss.pdf",
)
plot_metric_all_models(
    models_citeseer, "mrr",
    title="Citeseer – GCN: MRR vs ArScore (All Models)",
    savepath="/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot/citeseer_all_models_mrr2.pdf",
)
plot_metric_all_models(
    models_citeseer, "auc",
    title="Citeseer – GCN: AUC vs ArScore (All Models)",
    savepath="/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot/citeseer_all_models_auc2.pdf",
)
plot_metric_all_models(
    models_citeseer, "ap",
    title="Citeseer – GCN: AP vs ArScore (All Models)",
    savepath="/hkfs/work/workspace/scratch/cc7738-automorphism/ANP4Link/syn_real/exp_plot/citeseer_all_models_ap2.pdf",
)
