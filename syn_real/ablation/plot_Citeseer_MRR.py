data = """
Metric,Hits@1,MRR,AUC,AP
Citeseer_GCN_inter0.00_intra0.00_total0_Orbits_658.00_Norm_0.81_ArScore_0.90,26.52 ± 14.78,48.61 ± 13.29,99.65 ± 0.08,99.59 ± 0.11
Citeseer_GCN_inter0.10_intra0.50_total200_Orbits_1192.00_Norm_0.80_ArScore_0.82,34.79 ± 13.40,48.34 ± 10.77,99.50 ± 0.14,99.41 ± 0.17
Citeseer_GCN_inter0.10_intra0.50_total1000_Orbits_1849.00_Norm_0.77_ArScore_0.72,31.91 ± 11.93,49.84 ± 4.78,99.42 ± 0.16,99.35 ± 0.14
Citeseer_GCN_inter0.10_intra0.50_total2000_Orbits_2558.00_Norm_0.73_ArScore_0.62,26.95 ± 13.41,39.74 ± 9.66,98.92 ± 0.24,98.86 ± 0.15
Citeseer_GCN_inter0.10_intra0.50_total3000_Orbits_3221.00_Norm_0.69_ArScore_0.52,31.82 ± 14.21,43.70 ± 10.66,98.61 ± 0.33,98.61 ± 0.34
Citeseer_GCN_inter0.10_intra0.50_total4000_Orbits_3776.00_Norm_0.65_ArScore_0.43,4.97 ± 4.24,20.78 ± 3.08,98.03 ± 0.22,97.82 ± 0.29
Citeseer_GCN_inter0.10_intra0.50_total5000_Orbits_4338.00_Norm_0.61_ArScore_0.35,16.60 ± 10.42,27.99 ± 7.70,97.65 ± 0.47,97.55 ± 0.48
Citeseer_GCN_inter0.10_intra0.50_total7000_Orbits_5131.00_Norm_0.53_ArScore_0.23,5.12 ± 3.29,13.18 ± 2.00,96.32 ± 0.56,95.86 ± 0.50
Citeseer_GCN_inter0.10_intra0.50_total8000_Orbits_5398.00_Norm_0.50_ArScore_0.19,13.08 ± 4.90,21.32 ± 4.65,96.49 ± 0.74,96.27 ± 0.92
Citeseer_GCN_inter0.10_intra0.50_total10000_Orbits_5847.00_Norm_0.42_ArScore_0.12,10.95 ± 5.65,18.08 ± 3.87,95.03 ± 1.60,94.90 ± 1.64
Citeseer_GCN_inter0.10_intra0.50_total14000_Orbits_6265.00_Norm_0.28_ArScore_0.06,5.61 ± 3.17,10.56 ± 2.93,91.81 ± 2.05,91.28 ± 2.01
Citeseer_GIN_inter0.00_intra0.00_total0_Orbits_658.00_Norm_0.81_ArScore_0.90,13.16 ± 7.53,20.37 ± 3.99,95.43 ± 1.10,94.73 ± 1.34
Citeseer_GIN_inter0.10_intra0.50_total200_Orbits_1192.00_Norm_0.80_ArScore_0.82,3.14 ± 3.52,14.17 ± 3.56,95.40 ± 0.71,94.69 ± 0.61
Citeseer_GIN_inter0.10_intra0.50_total1000_Orbits_1849.00_Norm_0.77_ArScore_0.72,8.35 ± 8.19,18.09 ± 4.48,94.69 ± 1.14,94.08 ± 1.11
Citeseer_GIN_inter0.10_intra0.50_total2000_Orbits_2558.00_Norm_0.73_ArScore_0.62,1.52 ± 3.50,11.37 ± 2.29,92.73 ± 0.79,92.24 ± 0.50
Citeseer_GIN_inter0.10_intra0.50_total3000_Orbits_3221.00_Norm_0.69_ArScore_0.52,2.17 ± 1.76,10.79 ± 2.36,90.53 ± 1.34,90.29 ± 1.07
Citeseer_GIN_inter0.10_intra0.50_total4000_Orbits_3776.00_Norm_0.65_ArScore_0.43,0.82 ± 1.56,10.45 ± 1.82,89.78 ± 1.16,89.30 ± 1.16
Citeseer_GIN_inter0.10_intra0.50_total5000_Orbits_4338.00_Norm_0.61_ArScore_0.35,13.17 ± 2.80,15.77 ± 2.23,86.46 ± 2.00,86.91 ± 1.72
Citeseer_GIN_inter0.10_intra0.50_total7000_Orbits_5131.00_Norm_0.53_ArScore_0.23,3.80 ± 3.64,7.51 ± 3.36,78.42 ± 10.10,78.34 ± 10.08
Citeseer_GIN_inter0.10_intra0.50_total8000_Orbits_5398.00_Norm_0.50_ArScore_0.19,5.84 ± 4.45,9.61 ± 3.08,82.29 ± 1.19,82.59 ± 0.82
Citeseer_GIN_inter0.10_intra0.50_total10000_Orbits_5847.00_Norm_0.42_ArScore_0.12,2.48 ± 2.45,5.91 ± 2.72,75.34 ± 9.54,75.68 ± 9.45
Citeseer_GIN_inter0.10_intra0.50_total14000_Orbits_6265.00_Norm_0.28_ArScore_0.06,1.38 ± 1.53,2.82 ± 2.89,61.25 ± 12.04,61.68 ± 12.43
Citeseer_SAGE_inter0.00_intra0.00_total0_Orbits_658.00_Norm_0.81_ArScore_0.90,8.57 ± 10.35,20.28 ± 12.22,96.19 ± 2.00,96.10 ± 1.76
Citeseer_SAGE_inter0.10_intra0.50_total200_Orbits_1192.00_Norm_0.80_ArScore_0.82,13.35 ± 17.54,24.01 ± 15.47,95.99 ± 1.77,95.95 ± 2.00
Citeseer_SAGE_inter0.10_intra0.50_total1000_Orbits_1849.00_Norm_0.77_ArScore_0.72,7.07 ± 4.38,15.82 ± 5.18,94.18 ± 1.73,94.48 ± 1.51
Citeseer_SAGE_inter0.10_intra0.50_total2000_Orbits_2558.00_Norm_0.73_ArScore_0.62,6.93 ± 5.14,20.13 ± 10.96,93.26 ± 3.49,93.49 ± 3.24
Citeseer_SAGE_inter0.10_intra0.50_total3000_Orbits_3221.00_Norm_0.69_ArScore_0.52,9.68 ± 6.51,16.45 ± 4.91,91.69 ± 2.63,91.50 ± 3.32
Citeseer_SAGE_inter0.10_intra0.50_total4000_Orbits_3776.00_Norm_0.65_ArScore_0.43,4.63 ± 4.00,12.57 ± 5.13,91.81 ± 2.91,91.79 ± 2.97
Citeseer_SAGE_inter0.10_intra0.50_total5000_Orbits_4338.00_Norm_0.61_ArScore_0.35,4.90 ± 2.17,10.34 ± 3.02,86.72 ± 4.49,87.18 ± 4.51
Citeseer_SAGE_inter0.10_intra0.50_total7000_Orbits_5131.00_Norm_0.53_ArScore_0.23,5.67 ± 4.24,10.56 ± 5.06,85.62 ± 5.20,85.73 ± 5.65
Citeseer_SAGE_inter0.10_intra0.50_total8000_Orbits_5398.00_Norm_0.50_ArScore_0.19,5.10 ± 3.59,10.61 ± 5.20,82.41 ± 6.02,83.26 ± 5.81
Citeseer_SAGE_inter0.10_intra0.50_total10000_Orbits_5847.00_Norm_0.42_ArScore_0.12,2.68 ± 1.52,6.89 ± 2.12,77.59 ± 3.97,78.44 ± 4.14
Citeseer_SAGE_inter0.10_intra0.50_total14000_Orbits_6265.00_Norm_0.28_ArScore_0.06,1.53 ± 1.27,5.16 ± 2.18,74.68 ± 5.96,75.60 ± 5.73
Citeseer_ChebGCN_inter0.00_intra0.00_total0_Orbits_658.00_Norm_0.81_ArScore_0.90,14.20 ± 11.71,27.07 ± 10.19,97.59 ± 0.46,97.45 ± 0.47
Citeseer_ChebGCN_inter0.10_intra0.50_total200_Orbits_1192.00_Norm_0.80_ArScore_0.82,14.65 ± 6.99,28.40 ± 6.49,97.76 ± 0.41,97.72 ± 0.32
Citeseer_ChebGCN_inter0.10_intra0.50_total1000_Orbits_1849.00_Norm_0.77_ArScore_0.72,16.95 ± 6.44,27.45 ± 5.14,97.20 ± 0.44,97.21 ± 0.46
Citeseer_ChebGCN_inter0.10_intra0.50_total2000_Orbits_2558.00_Norm_0.73_ArScore_0.62,13.97 ± 7.49,23.05 ± 3.27,94.90 ± 0.58,94.88 ± 0.48
Citeseer_ChebGCN_inter0.10_intra0.50_total3000_Orbits_3221.00_Norm_0.69_ArScore_0.52,11.23 ± 5.26,22.76 ± 4.13,92.50 ± 1.31,92.83 ± 1.09
Citeseer_ChebGCN_inter0.10_intra0.50_total4000_Orbits_3776.00_Norm_0.65_ArScore_0.43,5.17 ± 4.03,12.33 ± 3.00,88.75 ± 1.86,89.36 ± 1.64
Citeseer_ChebGCN_inter0.10_intra0.50_total5000_Orbits_4338.00_Norm_0.61_ArScore_0.35,9.22 ± 6.04,15.23 ± 4.23,84.72 ± 1.89,86.03 ± 1.19
Citeseer_ChebGCN_inter0.10_intra0.50_total7000_Orbits_5131.00_Norm_0.53_ArScore_0.23,4.87 ± 2.62,8.83 ± 2.23,79.21 ± 1.42,80.87 ± 1.26
Citeseer_ChebGCN_inter0.10_intra0.50_total8000_Orbits_5398.00_Norm_0.50_ArScore_0.19,4.33 ± 3.14,7.57 ± 2.49,76.92 ± 1.49,78.45 ± 1.67
Citeseer_ChebGCN_inter0.10_intra0.50_total10000_Orbits_5847.00_Norm_0.42_ArScore_0.12,3.75 ± 1.26,7.09 ± 0.94,72.16 ± 2.07,74.41 ± 2.13
Citeseer_ChebGCN_inter0.10_intra0.50_total14000_Orbits_6265.00_Norm_0.28_ArScore_0.06,2.67 ± 0.81,4.81 ± 0.62,68.81 ± 2.76,70.88 ± 2.37
Citeseer_GAT_inter0.00_intra0.00_total0_Orbits_658.00_Norm_0.81_ArScore_0.90,14.15 ± 8.08,31.93 ± 8.23,99.42 ± 0.14,99.22 ± 0.24
Citeseer_GAT_inter0.10_intra0.50_total200_Orbits_1192.00_Norm_0.80_ArScore_0.82,11.14 ± 5.72,21.16 ± 4.89,98.94 ± 0.32,98.60 ± 0.23
Citeseer_GAT_inter0.10_intra0.50_total1000_Orbits_1849.00_Norm_0.77_ArScore_0.72,6.63 ± 6.38,25.82 ± 4.77,99.17 ± 0.14,98.96 ± 0.13
Citeseer_GAT_inter0.10_intra0.50_total2000_Orbits_2558.00_Norm_0.73_ArScore_0.62,14.28 ± 6.38,24.61 ± 6.22,98.28 ± 0.71,98.02 ± 0.63
Citeseer_GAT_inter0.10_intra0.50_total3000_Orbits_3221.00_Norm_0.69_ArScore_0.52,9.68 ± 7.86,19.10 ± 4.48,98.31 ± 0.74,98.01 ± 0.80
Citeseer_GAT_inter0.10_intra0.50_total4000_Orbits_3776.00_Norm_0.65_ArScore_0.43,13.99 ± 4.96,27.72 ± 7.61,97.89 ± 1.07,97.71 ± 0.92
Citeseer_GAT_inter0.10_intra0.50_total5000_Orbits_4338.00_Norm_0.61_ArScore_0.35,13.77 ± 7.58,23.06 ± 3.30,97.81 ± 0.41,97.56 ± 0.45
Citeseer_GAT_inter0.10_intra0.50_total7000_Orbits_5131.00_Norm_0.53_ArScore_0.23,10.80 ± 6.37,21.82 ± 8.21,96.56 ± 1.26,96.35 ± 1.09
Citeseer_GAT_inter0.10_intra0.50_total8000_Orbits_5398.00_Norm_0.50_ArScore_0.19,5.60 ± 2.29,14.75 ± 2.74,95.87 ± 1.41,95.42 ± 1.14
Citeseer_GAT_inter0.10_intra0.50_total10000_Orbits_5847.00_Norm_0.42_ArScore_0.12,6.94 ± 4.10,13.61 ± 5.15,93.11 ± 4.54,93.16 ± 4.25
Citeseer_GAT_inter0.10_intra0.50_total14000_Orbits_6265.00_Norm_0.28_ArScore_0.06,5.56 ± 4.07,10.22 ± 1.75,91.93 ± 2.09,91.73 ± 1.93
"""

import seaborn as sns
models = ["GCN", "GAT", "GIN", "GraphSAGE", "MixHopGCN", "ChebGCN", "LINKX"]
set3_colors = sns.color_palette("Set3", len(models))

# Assign colors to models
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from collections import defaultdict



raw_data = [
    ("GCN",
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
     [49.84, 48.61, 48.34, 43.7, 39.74, 27.99, 21.32, 20.78, 18.08, 13.18, 10.56],
     [4.78, 13.29, 10.77, 10.66, 9.66, 7.7, 4.65, 3.08, 3.87, 2.0, 2.93]),
    # ("GAT",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [31.93, 27.72, 25.82, 24.61, 23.06, 21.82, 21.16, 19.1, 14.75, 13.61, 10.22],
    #  [8.23, 7.61, 4.77, 6.22, 3.3, 8.21, 4.89, 4.48, 2.74, 5.15, 1.75]),
    # ("GIN",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [20.37, 18.09, 15.77, 14.17, 11.37, 10.79, 10.45, 9.61, 7.51, 5.91, 2.82],
    #  [3.99, 4.48, 2.23, 3.56, 2.29, 2.36, 1.82, 3.08, 3.36, 2.72, 2.89]),
    # ("GraphSAGE",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [24.01, 20.28, 20.13, 16.45, 15.82, 12.57, 10.61, 10.56, 10.34, 6.89, 5.16],
    #  [15.47, 12.22, 10.96, 4.91, 5.18, 5.13, 5.2, 5.06, 3.02, 2.12, 2.18]),
    # ("LINKX",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [61.82, 56.8, 51.21, 50.44, 44.59, 39.72, 36.21, 32.68, 32.09, 18.6, 12.34],
    #  [5.53, 7.94, 6.15, 15.06, 15.02, 9.74, 4.3, 8.6, 12.39, 2.41, 4.56]),
    # ("MixHopGCN",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [70.99, 52.17, 51.84, 43.61, 39.02, 36.87, 36.12, 35.69, 34.84, 25.08, 14.32],
    #  [27.71, 4.79, 9.3, 6.49, 2.52, 10.14, 4.8, 9.47, 6.12, 11.08, 2.42]),
    # #  #bs1024_lr0.02_testbs2048
    # ("BUDDY",
    # [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19, 0.12, 0.06],
    # [29.06, 31.40, 26.07, 37.07, 40.97, 38.80, 22.80, 37.20, 32.22, 35.62, 34.86],
    # [7.08, 4.33, 8.50, 5.59, 4.09, 1.98, 4.00, 3.34, 3.14, 5.89, 5.28]),
    # ("ChebGCN",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [28.4, 27.45, 27.07, 23.05, 22.76, 15.23, 12.33, 8.83, 7.57, 7.09, 4.81],
    #  [6.49, 5.14, 10.19, 3.27, 4.13, 4.23, 3.0, 2.23, 2.49, 0.94, 0.62]),
    ("Proposed w.o. D1",
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
     [83.4, 71.65, 69.35, 67.94, 64.67, 64.18, 63.9, 43.61, 41.19, 41.19, 36.42],
     [11.4, 12.54, 12.85, 12.46, 8.23, 7.58, 9.91, 5.35, 3.23, 3.23, 9.45]),
    # ("Proposed w.o D2",
    #  [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
    #  [79.99, 76.28, 72.2, 66.62, 66.62, 66.62, 58.34, 55.88, 49.56, 41.81, 38.42],
    #  [12.23, 7.76, 16.27, 12.13, 12.13, 12.13, 13.8, 12.07, 10.0, 7.54, 12.32]),
    ("Proposed w.o Dropout",
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
     [71.82, 66.8, 61.21, 60.44, 54.59, 49.72, 46.21, 42.68, 42.09, 28.6, 22.34],
     [12.23, 7.76, 16.27, 12.13, 12.13, 12.13, 13.8, 12.07, 10.0, 7.54, 12.32]),

    ("Proposed",
     [0.9, 0.82, 0.72, 0.62, 0.52, 0.43, 0.35, 0.23, 0.19,  0.12, 0.06],
     [85.72, 83.02, 74.85, 72.06, 68.64, 67.42, 66.61, 65.97, 55.19,  42.59, 41.19],
     [5.57, 6.87, 7.6, 5.95, 4.59, 10.02, 16.27, 11.65, 12.1, 8.04, 3.23]),
]


# Assuming interpolated_data already exists
colors = plt.cm.get_cmap('tab10', len(raw_data))

new_alpha = np.arange(0.1, 1.0, 0.2)
interpolated_data = defaultdict(dict)

TITLE_SIZE = 26
LABEL_SIZE = 35
TICK_SIZE = 35
LEGEND_SIZE = 26
LEGEND_TITLE_SIZE = 24
ANNOTATION_SIZE = 24
FIGSIZE = (10, 8)
DPI = 300
LEGENG_SIZE = 20

# Perform interpolation for each model
for model, alpha, best_valid, variance in raw_data:
    interpolated_data[model]["alpha"] =  [1-i for i in alpha] 
    interpolated_data[model]["best_valid"] = best_valid
    interpolated_data[model]["variance"] = variance
    
fig, ax = plt.subplots(figsize=(10, 6))

dashed_models = {"ChebGCN", "LINKX", "GIN"}  # Models that will have dashed lines
line_styles = {model: "--" if model in dashed_models else "-" for model in interpolated_data.keys()}
alpha_values = {model: 0.5 if model in dashed_models else 1.0 for model in interpolated_data.keys()}  # Reduce opacity for dashed lines


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
        fmt='o',
        color=color,
        alpha=0.3, 
        capsize=6,
        elinewidth=2,
        capthick=2
    )

# Legend
legend = ax.legend(
    fontsize=LEGEND_SIZE,
    loc="upper right",
    bbox_to_anchor=(0.95, 1),
    ncol=1,
    frameon=True,
    framealpha=0.5,
    fancybox=True
)
legend.get_frame().set_facecolor('white')

ax.set_xlabel(r"$\alpha_{\mathcal{V}}$", fontsize=LABEL_SIZE)
ax.set_ylabel("MRR (/%)", fontsize=LABEL_SIZE)
ax.set_xticks(new_alpha)
ax.set_yticks(np.arange(0, 101, 20))
ax.tick_params(axis='both', labelsize=TICK_SIZE)

plt.tight_layout()
plt.savefig('Ablation_Exp1_Citeseer_SYN_Real_MRR.pdf')
