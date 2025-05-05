import pandas as pd
import re
from io import StringIO

# Paste your CSV content here as a multiline string
csv_raw = """Metric,Hits@1,MRR,AUC,AP
Cora_ChebGCN_inter0.00_intra0.00_total0_Orbits_1007.00_Norm_0.72_ArScore_0.81,16.96 ± 4.08,28.71 ± 4.29,98.41 ± 0.13,98.23 ± 0.14
Cora_ChebGCN_inter0.10_intra0.50_total50_Orbits_1650.00_Norm_0.71_ArScore_0.70,16.31 ± 6.30,24.71 ± 5.39,98.19 ± 0.22,97.84 ± 0.26
Cora_ChebGCN_inter0.10_intra0.50_total250_Orbits_2023.00_Norm_0.70_ArScore_0.63,10.56 ± 5.43,21.38 ± 5.21,97.82 ± 0.21,97.40 ± 0.30
Cora_ChebGCN_inter0.10_intra0.50_total1000_Orbits_2754.00_Norm_0.66_ArScore_0.50,10.97 ± 8.54,21.01 ± 4.68,96.56 ± 0.79,96.60 ± 0.68
Cora_ChebGCN_inter0.10_intra0.50_total1000_Orbits_2754.00_Norm_0.66_ArScore_0.49,9.66 ± 6.57,19.82 ± 5.16,96.76 ± 0.32,96.57 ± 0.30
Cora_ChebGCN_inter0.10_intra0.50_total1750_Orbits_3228.00_Norm_0.63_ArScore_0.40,9.22 ± 5.85,15.35 ± 3.71,94.09 ± 0.56,93.74 ± 0.68
Cora_ChebGCN_inter0.10_intra0.50_total3000_Orbits_3834.00_Norm_0.56_ArScore_0.29,9.06 ± 6.54,16.75 ± 4.98,91.18 ± 1.02,91.71 ± 0.89
Cora_ChebGCN_inter0.10_intra0.50_total4500_Orbits_4380.00_Norm_0.49_ArScore_0.19,5.22 ± 3.03,10.26 ± 2.32,86.24 ± 1.77,86.93 ± 1.60
Cora_ChebGCN_inter0.10_intra0.50_total5000_Orbits_4520.00_Norm_0.46_ArScore_0.17,4.51 ± 2.72,9.77 ± 2.48,84.88 ± 1.42,85.93 ± 1.36
Cora_ChebGCN_inter0.10_intra0.50_total7000_Orbits_4862.00_Norm_0.38_ArScore_0.10,2.43 ± 1.40,5.33 ± 1.41,77.29 ± 3.20,79.17 ± 2.80


Cora_SAGE_inter0.00_intra0.00_total0_Orbits_1007.00_Norm_0.72_ArScore_0.81,13.99 ± 7.03,24.79 ± 8.27,97.91 ± 0.62,97.63 ± 0.86
Cora_SAGE_inter0.10_intra0.50_total50_Orbits_1650.00_Norm_0.71_ArScore_0.70,6.29 ± 4.48,16.30 ± 5.24,97.96 ± 0.57,97.35 ± 0.73
Cora_SAGE_inter0.10_intra0.50_total50_Orbits_1650.00_Norm_0.71_ArScore_0.70,6.29 ± 4.48,16.30 ± 5.24,97.96 ± 0.57,97.35 ± 0.73
Cora_SAGE_inter0.10_intra0.50_total250_Orbits_2023.00_Norm_0.70_ArScore_0.63,12.05 ± 5.81,24.20 ± 4.74,97.96 ± 0.60,97.71 ± 0.65
Cora_SAGE_inter0.10_intra0.50_total1000_Orbits_2754.00_Norm_0.66_ArScore_0.49,7.22 ± 5.41,17.53 ± 5.70,97.05 ± 0.85,96.66 ± 0.89
Cora_SAGE_inter0.10_intra0.50_total1750_Orbits_3228.00_Norm_0.63_ArScore_0.40,5.89 ± 4.15,13.66 ± 4.13,95.93 ± 0.93,95.60 ± 1.12
Cora_SAGE_inter0.10_intra0.50_total3000_Orbits_3834.00_Norm_0.56_ArScore_0.29,2.37 ± 1.69,8.94 ± 2.54,93.38 ± 0.99,92.85 ± 0.99
Cora_SAGE_inter0.10_intra0.50_total4500_Orbits_4380.00_Norm_0.49_ArScore_0.19,5.33 ± 2.39,12.26 ± 5.25,90.27 ± 1.93,90.59 ± 1.81
Cora_SAGE_inter0.10_intra0.50_total5000_Orbits_4520.00_Norm_0.46_ArScore_0.17,4.38 ± 3.62,11.01 ± 3.77,90.37 ± 2.24,90.62 ± 2.10
Cora_SAGE_inter0.10_intra0.50_total7000_Orbits_4862.00_Norm_0.38_ArScore_0.10,2.96 ± 2.20,8.26 ± 3.21,85.82 ± 2.03,86.14 ± 2.14

Cora_MixHopGCN_inter0.00_intra0.00_total0_Orbits_1007.00_Norm_0.72_ArScore_0.81,51.74 ± 25.45,65.87 ± 14.64,99.55 ± 0.17,99.56 ± 0.15
Cora_MixHopGCN_inter0.10_intra0.50_total50_Orbits_1650.00_Norm_0.71_ArScore_0.70,44.58 ± 18.14,58.62 ± 11.24,99.28 ± 0.09,99.35 ± 0.15
Cora_MixHopGCN_inter0.10_intra0.50_total250_Orbits_2023.00_Norm_0.70_ArScore_0.63,10.95 ± 23.63,17.65 ± 29.31,64.82 ± 23.86,64.80 ± 23.84
Cora_MixHopGCN_inter0.10_intra0.50_total1000_Orbits_2754.00_Norm_0.66_ArScore_0.17,0.00 ± 0.00,0.12 ± 0.00,50.00 ± 0.00,50.00 ± 0.00
Cora_MixHopGCN_inter0.10_intra0.50_total1750_Orbits_3228.00_Norm_0.63_ArScore_0.40,0.00 ± 0.00,0.12 ± 0.00,50.00 ± 0.00,50.00 ± 0.00
Cora_MixHopGCN_inter0.10_intra0.50_total3000_Orbits_3834.00_Norm_0.56_ArScore_0.29,0.00 ± 0.00,0.12 ± 0.00,50.00 ± 0.00,50.00 ± 0.00
Cora_MixHopGCN_inter0.10_intra0.50_total4500_Orbits_4380.00_Norm_0.49_ArScore_0.19,0.00 ± 0.00,0.12 ± 0.00,50.00 ± 0.00,50.00 ± 0.00
Cora_MixHopGCN_inter0.10_intra0.50_total5000_Orbits_4520.00_Norm_0.46_ArScore_0.17,0.00 ± 0.00,0.12 ± 0.00,50.00 ± 0.00,50.00 ± 0.00
Cora_MixHopGCN_inter0.10_intra0.50_total7000_Orbits_4862.00_Norm_0.38_ArScore_0.10,0.00 ± 0.00,0.12 ± 0.00,50.00 ± 0.00,50.00 ± 0.00


Cora_GIN_inter0.00_intra0.00_total0_Orbits_1007.00_Norm_0.72_ArScore_0.81,5.36 ± 4.57,10.88 ± 4.81,91.83 ± 14.70,90.88 ± 14.37
Cora_GIN_inter0.10_intra0.50_total50_Orbits_1650.00_Norm_0.71_ArScore_0.70,8.02 ± 5.14,15.57 ± 5.25,95.76 ± 0.78,94.64 ± 0.80
Cora_GIN_inter0.10_intra0.50_total250_Orbits_2023.00_Norm_0.70_ArScore_0.63,10.39 ± 2.23,15.80 ± 2.59,95.53 ± 0.59,94.69 ± 0.72
Cora_GIN_inter0.10_intra0.50_total1000_Orbits_2754.00_Norm_0.66_ArScore_0.49,3.96 ± 3.48,14.43 ± 3.99,94.91 ± 0.90,94.38 ± 0.85
Cora_GIN_inter0.10_intra0.50_total1750_Orbits_3228.00_Norm_0.63_ArScore_0.40,5.38 ± 3.49,9.19 ± 3.93,84.81 ± 13.14,84.47 ± 12.82
Cora_GIN_inter0.10_intra0.50_total3000_Orbits_3834.00_Norm_0.56_ArScore_0.29,1.78 ± 1.62,6.87 ± 4.00,82.49 ± 17.14,81.89 ± 16.83
Cora_GIN_inter0.10_intra0.50_total4500_Orbits_4380.00_Norm_0.49_ArScore_0.19,1.34 ± 1.63,4.06 ± 3.72,70.24 ± 17.78,70.18 ± 17.67
Cora_GIN_inter0.10_intra0.50_total5000_Orbits_4520.00_Norm_0.46_ArScore_0.17,1.76 ± 1.82,4.42 ± 3.89,72.01 ± 18.96,71.82 ± 18.78
Cora_GIN_inter0.10_intra0.50_total7000_Orbits_4862.00_Norm_0.38_ArScore_0.10,1.08 ± 1.68,3.07 ± 3.30,65.24 ± 16.38,65.45 ± 16.53


Cora_GCN_inter0.00_intra0.00_total0_Orbits_1007.00_Norm_0.72_ArScore_0.81,32.21 ± 13.26,42.28 ± 11.15,99.41 ± 0.23,99.28 ± 0.28
Cora_GCN_inter0.10_intra0.50_total50_Orbits_1650.00_Norm_0.71_ArScore_0.70,17.72 ± 7.71,34.09 ± 8.75,99.39 ± 0.09,99.18 ± 0.20
Cora_GCN_inter0.10_intra0.50_total250_Orbits_2023.00_Norm_0.70_ArScore_0.63,13.11 ± 8.24,27.06 ± 5.83,99.06 ± 0.22,98.81 ± 0.27
Cora_GCN_inter0.10_intra0.50_total1000_Orbits_2754.00_Norm_0.66_ArScore_0.49,4.72 ± 3.09,18.37 ± 3.90,98.51 ± 0.35,98.23 ± 0.39
Cora_GCN_inter0.10_intra0.50_total1750_Orbits_3228.00_Norm_0.63_ArScore_0.40,6.88 ± 3.24,20.83 ± 3.29,98.01 ± 0.45,97.75 ± 0.48
Cora_GCN_inter0.10_intra0.50_total3000_Orbits_3834.00_Norm_0.56_ArScore_0.29,3.99 ± 2.84,14.78 ± 4.48,97.27 ± 0.42,96.97 ± 0.44
Cora_GCN_inter0.10_intra0.50_total4500_Orbits_4380.00_Norm_0.49_ArScore_0.19,13.68 ± 6.56,21.29 ± 6.17,96.45 ± 0.57,96.43 ± 0.61
Cora_GCN_inter0.10_intra0.50_total5000_Orbits_4520.00_Norm_0.46_ArScore_0.17,5.14 ± 2.53,16.50 ± 3.50,96.29 ± 0.67,95.96 ± 0.76
Cora_GCN_inter0.10_intra0.50_total7000_Orbits_4889.00_Norm_0.36_ArScore_0.10,7.41 ± 4.41,12.82 ± 3.91,94.07 ± 0.86,93.91 ± 0.92

Cora_GAT_inter0.00_intra0.00_total0_Orbits_1007.00_Norm_0.72_ArScore_0.81,16.41 ± 8.36,30.45 ± 6.74,99.28 ± 0.16,98.98 ± 0.25
Cora_GAT_inter0.10_intra0.50_total50_Orbits_1650.00_Norm_0.71_ArScore_0.70,8.79 ± 5.38,20.28 ± 5.60,98.97 ± 0.23,98.46 ± 0.44
Cora_GAT_inter0.10_intra0.50_total250_Orbits_2023.00_Norm_0.70_ArScore_0.63,9.28 ± 7.50,25.24 ± 7.63,98.99 ± 0.13,98.57 ± 0.20
Cora_GAT_inter0.10_intra0.50_total1000_Orbits_2754.00_Norm_0.66_ArScore_0.49,9.72 ± 9.69,19.74 ± 9.83,98.75 ± 0.20,98.24 ± 0.26
Cora_GAT_inter0.10_intra0.50_total1750_Orbits_3228.00_Norm_0.63_ArScore_0.40,7.95 ± 5.48,21.77 ± 7.00,98.48 ± 0.25,98.15 ± 0.40
Cora_GAT_inter0.10_intra0.50_total3000_Orbits_3834.00_Norm_0.56_ArScore_0.29,5.98 ± 4.85,14.80 ± 4.85,97.98 ± 0.22,97.51 ± 0.37
Cora_GAT_inter0.10_intra0.50_total4500_Orbits_4380.00_Norm_0.49_ArScore_0.19,4.71 ± 2.22,15.60 ± 3.41,97.72 ± 0.50,97.41 ± 0.57
Cora_GAT_inter0.10_intra0.50_total5000_Orbits_4520.00_Norm_0.46_ArScore_0.17,7.54 ± 5.54,17.06 ± 6.54,97.51 ± 0.50,97.11 ± 0.38
Cora_GAT_inter0.10_intra0.50_total7000_Orbits_4862.00_Norm_0.38_ArScore_0.10,4.68 ± 3.07,13.28 ± 3.10,96.42 ± 0.39,95.77 ± 0.56

Cora_LINKX_inter0.00_intra0.00_total0_Orbits_1007.00_Norm_0.72_ArScore_0.81,11.50 ± 8.36,18.60 ± 0.00,97.30 ± 0.32,97.05 ± 0.28
Cora_LINKX_inter0.10_intra0.50_total50_Orbits_1407.00_Norm_0.71_ArScore_0.74,22.22 ± 9.32,32.99 ± 9.10,98.03 ± 0.15,98.04 ± 0.13
Cora_LINKX_inter0.10_intra0.50_total50_Orbits_1657.00_Norm_0.71_ArScore_0.69,0.00 ± 5.89,11.67 ± 3.83,96.99 ± 0.18,96.84 ± 0.07
Cora_LINKX_inter0.10_intra0.50_total250_Orbits_2080.00_Norm_0.70_ArScore_0.62,8.05 ± 0.00,19.90 ± 0.00,96.81 ± 0.00,96.82 ± 0.00
Cora_LINKX_inter0.10_intra0.50_total1000_Orbits_2756.00_Norm_0.66_ArScore_0.49,10.43 ± 0.00,23.91 ± 0.00,94.28 ± 0.00,94.85 ± 0.00
Cora_LINKX_inter0.10_intra0.50_total1750_Orbits_3217.00_Norm_0.63_ArScore_0.41,16.02 ± 0.00,17.44 ± 0.00,92.82 ± 0.00,94.03 ± 0.00
Cora_LINKX_inter0.10_intra0.50_total3000_Orbits_3769.00_Norm_0.58_ArScore_0.30,6.69 ± 0.00,14.99 ± 0.00,89.63 ± 0.00,91.42 ± 0.00
Cora_LINKX_inter0.10_intra0.50_total4500_Orbits_4317.00_Norm_0.50_ArScore_0.20,1.42 ± 0.00,8.08 ± 0.00,86.69 ± 0.00,88.81 ± 0.00
Cora_LINKX_inter0.10_intra0.50_total5000_Orbits_4454.00_Norm_0.48_ArScore_0.18,6.48 ± 0.00,15.08 ± 0.00,85.47 ± 0.00,87.89 ± 0.00
Cora_LINKX_inter0.10_intra0.50_total7000_Orbits_4807.00_Norm_0.39_ArScore_0.11,6.22 ± 0.00,10.86 ± 0.00,83.48 ± 0.00,86.19 ± 0.00

# D1
Cora_D1_inter0.00_ArScore_0.19gcn_cn1_inter0.00_intra0.00_total0_drop_0.0_use_wl_False,21.52 ± 14.03,45.59 ± 11.42,99.76 ± 0.07,99.65 ± 0.09
Cora_D1_inter0.10ArScore_0.19gcn_cn1_inter0.00_intra0.00_total0_drop_0.0_use_wl_False,12.95 ± 9.73,47.36 ± 12.70,99.74 ± 0.07,99.61 ± 0.08
Cora_D1_inter0.10_ArScore_0.30gcn_cn1_inter0.10_intra0.50_total50_drop_0.0_use_wl_False,40.71 ± 22.93,62.30 ± 10.21,99.77 ± 0.11,99.75 ± 0.11
Cora_D1_inter0.10_ArScore_0.31gcn_cn1_inter0.10_intra0.50_total250_drop_0.0_use_wl_False,29.42 ± 9.62,45.83 ± 5.32,99.76 ± 0.05,99.64 ± 0.05
Cora_D1_inter0.10_ArScore_0.31gcn_cn1_inter0.10_intra0.50_total1000_drop_0.0_use_wl_False,44.47 ± 32.70,62.64 ± 20.27,99.83 ± 0.10,99.82 ± 0.10
Cora_D1_inter0.10_ArScore_0.29gcn_cn1_inter0.10_intra0.50_total1750_drop_0.0_use_wl_False,41.60 ± 17.54,57.83 ± 12.52,99.67 ± 0.12,99.63 ± 0.08
Cora_D1_inter0.10_ArScore_0.24gcn_cn1_inter0.10_intra0.50_total3000_drop_0.0_use_wl_False,41.11 ± 13.51,52.14 ± 3.52,99.63 ± 0.07,99.59 ± 0.07
Cora_D1_inter0.10_ArScore_0.19gcn_cn1_inter0.10_intra0.50_total4500_drop_0.0_use_wl_False,45.64 ± 14.15,65.46 ± 4.98,99.54 ± 0.30,99.69 ± 0.16
Cora_D1_inter0.10_ArScore_0.17gcn_cn1_inter0.10_intra0.50_total5000_drop_0.0_use_wl_False,70.67 ± 21.39,80.43 ± 12.34,99.62 ± 0.22,99.66 ± 0.19
Cora_D1_inter0.10_ArScore_0.09gcn_cn1_inter0.10_intra0.50_total7000_drop_0.0_use_wl_False,88.92 ± 10.93,85.53 ± 14.99,99.89 ± 0.08,99.90 ± 0.09

# D2
Cora_D2_inter1007.00_ArScore_0.19gcn_cn1_inter0.00_intra0.00_total0_drop_0.1_use_wl_False,24.30 ± 8.09,45.12 ± 5.11,99.80 ± 0.05,99.72 ± 0.08
Cora_D2_inter1605.00_ArScore_0.30gcn_cn1_inter0.10_intra0.50_total50_drop_0.1_use_wl_False,27.18 ± 27.29,35.64 ± 4.88,99.75 ± 0.03,99.68 ± 0.05
Cora_D2_inter1705.00_ArScore_0.31gcn_cn1_inter0.10_intra0.50_total250_drop_0.1_use_wl_False,5.70 ± 4.31,32.47 ± 7.89,99.71 ± 0.03,99.49 ± 0.19
Cora_D2_inter1704.00_ArScore_0.31gcn_cn1_inter0.10_intra0.50_total1000_drop_0.1_use_wl_False,5.94 ± 4.94,38.35 ± 4.47,99.73 ± 0.04,99.57 ± 0.09
Cora_D2_inter1578.00_ArScore_0.29gcn_cn1_inter0.10_intra0.50_total1750_drop_0.1_use_wl_False,12.57 ± 5.07,54.20 ± 17.44,99.79 ± 0.07,99.69 ± 0.27
Cora_D2_inter1320.00_ArScore_0.24gcn_cn1_inter0.10_intra0.50_total3000_drop_0.1_use_wl_False,25.35 ± 11.03,49.43 ± 10.93,99.74 ± 0.07,99.69 ± 0.13
Cora_D2_inter1013.00_ArScore_0.19gcn_cn1_inter0.10_intra0.50_total4500_drop_0.1_use_wl_False,31.11 ± 4.00,49.86 ± 7.44,99.78 ± 0.12,99.66 ± 0.23
Cora_D2_inter1098.00_ArScore_0.17gcn_cn1_inter0.10_intra0.50_total5000_drop_0.1_use_wl_False,37.96 ± 11.98,58.13 ± 11.50,99.68 ± 0.21,99.63 ± 0.21
Cora_D2_inter479.00_ArScore_0.09gcn_cn1_inter0.10_intra0.50_total7000_drop_0.1_use_wl_False,96.34 ± 2.86,97.42 ± 1.74,99.90 ± 0.05,99.86 ± 0.06

# D3
Cora_D3_inter1007.00_ArScore_0.19gcn_cn1_inter0.00_intra0.00_total0_drop_0.1_use_wl_True,20.85 ± 15.60,41.64 ± 13.77,99.73 ± 0.08,99.60 ± 0.06
Cora_D3_inter1605.00_ArScore_0.30gcn_cn1_inter0.10_intra0.50_total50_drop_0.1_use_wl_True,0.38 ± 0.23,34.89 ± 3.30,99.74 ± 0.07,99.45 ± 0.13
Cora_D3_inter1705.00_ArScore_0.31gcn_cn1_inter0.10_intra0.50_total250_drop_0.1_use_wl_True,24.12 ± 4.06,49.77 ± 2.04,99.86 ± 0.03,99.83 ± 0.05
Cora_D3_inter1704.00_ArScore_0.31gcn_cn1_inter0.10_intra0.50_total1000_drop_0.1_use_wl_True,30.71 ± 33.92,66.53 ± 14.99,99.83 ± 0.07,99.85 ± 0.05
Cora_D3_inter1578.00_ArScore_0.29gcn_cn1_inter0.10_intra0.50_total1750_drop_0.1_use_wl_True,15.81 ± 10.05,38.49 ± 6.13,99.76 ± 0.03,99.70 ± 0.03
Cora_D3_inter1320.00_ArScore_0.24gcn_cn1_inter0.10_intra0.50_total3000_drop_0.1_use_wl_True,49.61 ± 32.81,64.51 ± 20.85,99.81 ± 0.00,99.75 ± 0.07
Cora_D3_inter1013.00_ArScore_0.19gcn_cn1_inter0.10_intra0.50_total4500_drop_0.1_use_wl_True,38.50 ± 16.57,60.10 ± 10.83,99.80 ± 0.05,99.82 ± 0.09
Cora_D3_inter1898.00_ArScore_0.17gcn_cn1_inter0.10_intra0.50_total5000_drop_0.1_use_wl_True,55.91 ± 25.48,84.41 ± 1.87,99.78 ± 0.04,99.79 ± 0.05
Cora_D3_inter1479.00_ArScore_0.09gcn_cn1_inter0.10_intra0.50_total7000_drop_0.1_use_wl_True,86.05 ± 12.11,90.78 ± 7.34,99.94 ± 0.03,99.93 ± 0.01

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
    # Extract model name (e.g., GCN, GIN, D1, D3, etc.)
    match = re.search(r"Cora_([A-Za-z0-9]+)", metric_str)
    model = match.group(1) if match else "Unknown"

    # Extract ArScore
    arscore_match = re.search(r"ArScore_([0-9.]+)", metric_str)
    arscore = float(arscore_match.group(1)) if arscore_match else None

    # Extract total number of added edges
    total_match = re.search(r"total(\d+)", metric_str)
    total_edges = int(total_match.group(1)) if total_match else 0  # Default to 0 if not found

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
output_path = "/hkfs/work/workspace/scratch/cc7738-rebuttal/ANP4Link/syn_real/results/syn_cora.csv"

with open(output_path, "w") as f:
    f.write("raw_data = [\n")
    for model, ars, means, stds in raw_data:
        f.write(f'    ("{model}",\n')
        f.write(f"     {ars},\n")
        f.write(f"     {means},\n")
        f.write(f"     {stds}),\n")
    f.write("]\n")
