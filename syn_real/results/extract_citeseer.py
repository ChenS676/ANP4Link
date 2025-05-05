import pandas as pd
import re
from io import StringIO
import numpy as np
# Paste your CSV content here as a multiline string
csv_raw = """Metric,Hits@1,MRR,AUC,AP
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

Citeseer_LINKX_inter0.00_intra0.00_total0_Orbits_658.00_Norm_0.81_ArScore_0.10,13.60 ± 13.72,32.09 ± 12.39,98.46 ± 0.34,98.40 ± 0.37
Citeseer_LINKX_inter0.10_intra0.50_total200_Orbits_1193.00_Norm_0.80_ArScore_0.82,22.42 ± 12.82,39.72 ± 9.74,98.34 ± 0.08,98.47 ± 0.14
Citeseer_LINKX_inter0.10_intra0.50_total1000_Orbits_1037.00_Norm_0.82_ArScore_0.16,29.70 ± 25.57,50.44 ± 15.06,98.10 ± 0.36,98.40 ± 0.50
Citeseer_LINKX_inter0.10_intra0.50_total2000_Orbits_892.00_Norm_0.84_ArScore_0.13,27.58 ± 17.97,44.59 ± 15.02,97.54 ± 0.21,98.05 ± 0.12
Citeseer_LINKX_inter0.10_intra0.50_total3000_Orbits_745.00_Norm_0.85_ArScore_0.11,39.93 ± 22.83,61.82 ± 5.53,95.63 ± 0.30,96.45 ± 0.22
Citeseer_LINKX_inter0.10_intra0.50_total4000_Orbits_590.00_Norm_0.87_ArScore_0.09,36.65 ± 20.13,56.80 ± 7.94,95.38 ± 0.12,96.37 ± 0.11
Citeseer_LINKX_inter0.10_intra0.50_total5000_Orbits_471.00_Norm_0.89_ArScore_0.07,15.41 ± 6.83,36.21 ± 4.30,95.46 ± 0.37,95.92 ± 0.16
Citeseer_LINKX_inter0.10_intra0.50_total7000_Orbits_196.00_Norm_0.93_ArScore_0.03,38.97 ± 4.56,51.21 ± 6.15,92.06 ± 0.80,93.69 ± 0.42
Citeseer_LINKX_inter0.10_intra0.50_total8000_Orbits_98.00_Norm_0.95_ArScore_0.01,22.81 ± 12.81,32.68 ± 8.60,86.74 ± 1.61,88.57 ± 1.04
Citeseer_LINKX_inter0.10_intra0.50_total10000_Orbits_4.00_Norm_1.00_ArScore_0.00,2.22 ± 3.85,18.60 ± 2.41,45.78 ± 7.32,52.84 ± 4.37


Citeseer_MixHopGCN_inter0.00_intra0.00_total0_Orbits_658.00_Norm_0.81_ArScore_0.90,22.57 ± 9.83,52.17 ± 4.79,99.69 ± 0.06,99.62 ± 0.08
Citeseer_MixHopGCN_inter0.10_intra0.50_total200_Orbits_1192.00_Norm_0.80_ArScore_0.82,60.68 ± 31.63,70.99 ± 27.71,94.41 ± 15.60,94.47 ± 15.62
Citeseer_MixHopGCN_inter0.10_intra0.50_total1000_Orbits_1849.00_Norm_0.77_ArScore_0.72,8.28 ± 7.23,34.84 ± 6.12,99.03 ± 0.25,98.77 ± 0.18
Citeseer_MixHopGCN_inter0.10_intra0.50_total2000_Orbits_2558.00_Norm_0.73_ArScore_0.62,4.91 ± 3.57,36.12 ± 4.80,98.77 ± 0.07,98.73 ± 0.11
Citeseer_MixHopGCN_inter0.10_intra0.50_total3000_Orbits_3221.00_Norm_0.69_ArScore_0.52,12.81 ± 5.36,39.02 ± 2.52,96.69 ± 0.38,97.32 ± 0.31
Citeseer_MixHopGCN_inter0.10_intra0.50_total4000_Orbits_3776.00_Norm_0.65_ArScore_0.43,14.77 ± 12.95,36.87 ± 10.14,97.99 ± 0.23,97.94 ± 0.17
Citeseer_MixHopGCN_inter0.10_intra0.50_total5000_Orbits_4338.00_Norm_0.61_ArScore_0.35,23.32 ± 9.09,43.61 ± 6.49,95.11 ± 0.79,96.21 ± 0.45
Citeseer_MixHopGCN_inter0.10_intra0.50_total7000_Orbits_5131.00_Norm_0.53_ArScore_0.23,42.54 ± 16.61,51.84 ± 9.30,95.94 ± 0.44,96.58 ± 0.28
Citeseer_MixHopGCN_inter0.10_intra0.50_total8000_Orbits_5398.00_Norm_0.50_ArScore_0.19,22.81 ± 14.84,35.69 ± 9.47,94.55 ± 0.52,95.32 ± 0.33
Citeseer_MixHopGCN_inter0.10_intra0.50_total10000_Orbits_5847.00_Norm_0.42_ArScore_0.12,12.23 ± 12.27,25.08 ± 11.08,91.21 ± 1.95,92.94 ± 1.29
Citeseer_MixHopGCN_inter0.10_intra0.50_total14000_Orbits_6265.00_Norm_0.28_ArScore_0.06,4.89 ± 3.34,14.32 ± 2.42,89.01 ± 1.85,90.83 ± 1.46

# w D1
Citeseer_D1_inter_Orbits_658.00_ArScore_0.10gcn_cn0_inter0.00_intra0.00_total0_drop_0.0_use_wl_False,51.38 ± 27.13,67.94 ± 12.46,99.91 ± 0.05,99.86 ± 0.08
Citeseer_D1_interOrbits_714.00_ArScore_0.11gcn_cn0_inter0.10_intra0.50_total20_drop_0.0_use_wl_False,40.06 ± 12.97,64.18 ± 7.58,99.90 ± 0.04,99.86 ± 0.06
Citeseer_D1_interOrbits_906.00_ArScore_0.14gcn_cn0_inter0.10_intra0.50_total100_drop_0.0_use_wl_False,52.96 ± 16.32,71.65 ± 12.54,99.87 ± 0.04,99.88 ± 0.07
Citeseer_D1_interOrbits_990.00_ArScore_0.15gcn_cn0_inter0.10_intra0.50_total200_drop_0.0_use_wl_False,76.17 ± 21.78,85.72 ± 11.40,99.93 ± 0.04,99.92 ± 0.06
Citeseer_D1_interOrbits_1038.00_ArScore_0.16gcn_cn0_inter0.10_intra0.50_total300_drop_0.0_use_wl_False,43.38 ± 13.45,64.67 ± 8.23,99.81 ± 0.08,99.82 ± 0.05
Citeseer_D1_interOrbits_1054.00_ArScore_0.16gcn_cn0_inter0.10_intra0.50_total400_drop_0.0_use_wl_False,47.75 ± 19.26,69.35 ± 12.85,99.88 ± 0.05,99.87 ± 0.06
Citeseer_D1_interOrbits_1047.00_ArScore_0.16gcn_cn0_inter0.10_intra0.50_total500_drop_0.0_use_wl_False,9.94 ± 11.65,36.42 ± 9.45,99.77 ± 0.05,99.49 ± 0.14
Citeseer_D1_interOrbits_1043.00_ArScore_0.16gcn_cn0_inter0.10_intra0.50_total700_drop_0.0_use_wl_False,14.57 ± 7.47,43.61 ± 5.35,99.77 ± 0.07,99.63 ± 0.09
Citeseer_D1_interOrbits_1043.00_ArScore_0.16gcn_cn0_inter0.10_intra0.50_total800_drop_0.0_use_wl_False,44.97 ± 15.73,63.90 ± 9.91,99.82 ± 0.06,99.81 ± 0.05
Citeseer_D1_interOrbits_1015.00_ArScore_0.15gcn_cn0_inter0.10_intra0.50_total1000_drop_0.0_use_wl_False,14.64 ± 7.86,41.19 ± 3.23,99.76 ± 0.04,99.63 ± 0.08
Citeseer_D1_interOrbits_1015.00_ArScore_0.15gcn_cn0_inter0.10_intra0.50_total14000_drop_0.0_use_wl_False,14.64 ± 7.86,41.19 ± 3.23,99.76 ± 0.04,99.63 ± 0.08


# D3
Citeseer_D3_interOrbits_658.00_ArScore_0.10gcn_cn1_inter0.00_intra0.00_total0_drop_0.1_use_wl_False,43.10 ± 17.79,67.42 ± 10.02,99.88 ± 0.07,99.89 ± 0.06
Citeseer_D3_inter_Orbits_826.00_ArScore_0.12gcn_cn1_inter0.10_intra0.50_total20_drop_0.1_use_wl_False,52.27 ± 8.50,68.64 ± 4.59,99.89 ± 0.03,99.87 ± 0.03
Citeseer_D3_inter_Orbits_974.00_ArScore_0.15gcn_cn1_inter0.10_intra0.50_total100_drop_0.1_use_wl_False,35.73 ± 18.93,49.56 ± 12.10,99.82 ± 0.06,99.76 ± 0.05
Citeseer_D3_inter_Orbits_945.00_ArScore_0.14gcn_cn1_inter0.10_intra0.50_total200_drop_0.1_use_wl_False,41.23 ± 26.13,65.97 ± 11.65,99.92 ± 0.01,99.91 ± 0.02
Citeseer_D3_inter_Orbits_1016.00_ArScore_0.15gcn_cn1_inter0.10_intra0.50_total300_drop_0.1_use_wl_False,44.95 ± 24.87,66.61 ± 16.27,99.92 ± 0.05,99.88 ± 0.08
Citeseer_D3_inter_Orbits_1036.00_ArScore_0.16gcn_cn1_inter0.10_intra0.50_total400_drop_0.1_use_wl_False,63.44 ± 13.37,83.40 ± 5.57,99.93 ± 0.02,99.93 ± 0.03
Citeseer_D3_inter_Orbits_983.00_ArScore_0.15gcn_cn1_inter0.10_intra0.50_total1500_drop_0.1_use_wl_False,66.07 ± 7.51,83.02 ± 6.87,99.85 ± 0.06,99.87 ± 0.05
Citeseer_D3_inter_Orbits_1074.00_ArScore_0.16gcn_cn1_inter0.10_intra0.50_total700_drop_0.1_use_wl_False,62.58 ± 10.94,74.85 ± 7.60,99.90 ± 0.03,99.89 ± 0.04
Citeseer_D3_inter_Orbits_1058.00_ArScore_0.16gcn_cn1_inter0.10_intra0.50_total800_drop_0.1_use_wl_False,15.62 ± 7.28,42.59 ± 8.04,99.75 ± 0.05,99.68 ± 0.09
Citeseer_D3_inter_Orbits_1037.00_ArScore_0.16gcn_cn1_inter0.10_intra0.50_total1000_drop_0.1_use_wl_False,52.36 ± 12.32,72.06 ± 5.95,99.88 ± 0.06,99.84 ± 0.05


# without D2 
Citeseer_D2_inter_Orbits_658.00_ArScore_0.10gcn_cn0_inter0.00_intra0.00_total0_drop_0.1_use_wl_False,50.86 ± 21.28,66.62 ± 12.13,99.89 ± 0.04,99.86 ± 0.07
Citeseer_D2_inter__Orbits_658.00_ArScore_0.10gcn_cn0_inter0.00_intra0.00_total20_drop_0.1_use_wl_False,50.86 ± 21.28,66.62 ± 12.13,99.89 ± 0.04,99.86 ± 0.07
Citeseer_D2_inter__Orbits_658.00_ArScore_0.10gcn_cn0_inter0.00_intra0.00_total100_drop_0.1_use_wl_False,50.86 ± 21.28,66.62 ± 12.13,99.89 ± 0.04,99.86 ± 0.07
Citeseer_D2_inter__Orbits_990.00_ArScore_0.15gcn_cn0_inter0.10_intra0.50_total200_drop_0.1_use_wl_False,63.15 ± 12.06,76.28 ± 7.76,99.90 ± 0.04,99.88 ± 0.04
Citeseer_D2_inter__Orbits_1015.00_ArScore_0.15gcn_cn0_inter0.10_intra0.50_total1000_drop_0.1_use_wl_False,35.81 ± 19.50,58.34 ± 13.80,99.83 ± 0.04,99.77 ± 0.10
Citeseer_D2_inter__Orbits_906.00_ArScore_0.14gcn_cn0_inter0.10_intra0.50_total2000_drop_0.1_use_wl_False,55.46 ± 27.35,72.20 ± 16.27,99.90 ± 0.05,99.87 ± 0.09
Citeseer_D2_inter__Orbits_730.00_ArScore_0.11gcn_cn0_inter0.10_intra0.50_total3000_drop_0.1_use_wl_False,17.32 ± 12.73,41.81 ± 7.54,99.72 ± 0.08,99.52 ± 0.16
Citeseer_D2_inter__Orbits_577.00_ArScore_0.09gcn_cn0_inter0.10_intra0.50_total4000_drop_0.1_use_wl_False,36.00 ± 15.37,55.88 ± 12.07,99.72 ± 0.08,99.65 ± 0.11
Citeseer_D2_inter__Orbits_433.00_ArScore_0.07gcn_cn0_inter0.10_intra0.50_total5000_drop_0.1_use_wl_False,64.55 ± 22.44,79.99 ± 12.23,99.91 ± 0.05,99.89 ± 0.06
Citeseer_D2_inter__Orbits_177.00_ArScore_0.03gcn_cn0_inter0.10_intra0.50_total7000_drop_0.1_use_wl_False,18.43 ± 21.65,55.19 ± 10.00,99.74 ± 0.08,99.41 ± 0.23
Citeseer_D3_inter_Orbits_1015.00_ArScore_0.15gcn_cn0_inter0.10_intra0.50_total14000_drop_0.1_use_wl_False,14.64 ± 7.86,41.19 ± 3.23,99.76 ± 0.04,99.63 ± 0.08

"""
# Clean up ellipses if you're copying the full CSV
csv_data = StringIO(csv_raw)
import pandas as pd
import re
from io import StringIO

# Load CSV data into DataFrame
csv_data = StringIO(csv_raw)
df = pd.read_csv(csv_data)

# Function to extract Model name, ArScore, and TotalEdges from the Metric string
def parse_metric(metric_str):
    # Extract model name between 'Citeseer_' and '_inter'
    match = re.search(r"Citeseer_([A-Za-z0-9]+)_inter", metric_str)
    model = match.group(1) if match else "Unknown"

    # Extract ArScore
    arscore_match = re.search(r"ArScore_([0-9.]+)", metric_str)
    arscore = float(arscore_match.group(1)) if arscore_match else None

    # Extract total edges
    total_match = re.search(r"total(\d+)", metric_str)
    total_edges = int(total_match.group(1)) if total_match else 0

    return pd.Series([model, arscore, total_edges])

# Apply the parsing function to create new columns
df[['Model', 'ArScore', 'TotalEdges']] = df['Metric'].apply(parse_metric)

# Reorder columns for clarity
cols = ['Model', 'ArScore', 'TotalEdges', 'Metric'] + [col for col in df.columns if col not in ['Model', 'ArScore', 'TotalEdges', 'Metric']]
df = df[cols]

# Parse mean values from metrics in the format "mean ± std"
def parse_mean(metric_str):
    try:
        return float(re.match(r"([\d.]+)", str(metric_str)).group(1))
    except:
        return None

metrics = ['Hits@1', 'MRR', 'AUC', 'AP']
for metric in metrics:
    df[f'{metric}_mean'] = df[metric].apply(parse_mean)

# Create a sorted dictionary for each metric by model
sorted_metrics_by_model = {}

for model in df['Model'].unique():
    model_df = df[df['Model'] == model].copy()
    model_metrics = {}
    for metric in metrics:
        sorted_df = model_df.sort_values(by=f'{metric}_mean', ascending=False)
        model_metrics[metric] = sorted_df[['ArScore', metric, f'{metric}_mean']].reset_index(drop=True)
    sorted_metrics_by_model[model] = model_metrics

# Prepare data for export
model_name_map = {
    'GCN': 'GCN',
    'GAT': 'GAT',
    'GIN': 'GIN',
    'SAGE': 'GraphSAGE',
    'LINKX': 'LINKX',
    'GIN': 'GIN',
    'MixHopGCN': 'MixHopGCN',
    'ChebGCN': 'ChebGCN',
    'D1': 'Proposed w.o. D1',
    'D2': 'Proposed w.o D2',
    'D3': 'Proposed',
}

raw_data = []

for model_key, display_name in model_name_map.items():
    print(model_key, display_name)
    
    model_df = sorted_metrics_by_model[model_key]
    means, stds = [], []
    for metric in ['MRR']:  # Replace or extend with more metrics if needed
        for val in model_df[metric][metric]:
            match = re.match(r"([\d.]+)\s*±\s*([\d.]+)", str(val))
            if match:
                means.append(float(match.group(1)))
                stds.append(float(match.group(2)))
            else:
                means.append(None)
                stds.append(None)
    ar_list = model_df['AUC']['ArScore'].tolist()
    raw_data.append((display_name, ar_list, means, stds))

# Save raw_data to a file
output_path = "/hkfs/work/workspace/scratch/cc7738-rebuttal/ANP4Link/syn_real/results/syn_citeseer.csv"


def downsample_list(lst, n):
    """Downsample list to n evenly spaced points"""
    if len(lst) <= n:
        return lst  # no downsampling needed
    idx = np.round(np.linspace(0, len(lst) - 1, n)).astype(int)
    return [lst[i] for i in idx]


with open(output_path, "w") as f:
    f.write("raw_data = [\n")
    for model, ars, means, stds in raw_data:
        f.write(f'    ("{model}",\n')
        f.write(f"     {ars},\n")
        f.write(f"     {means},\n")
        f.write(f"     {stds}),\n")
    f.write("]\n")
