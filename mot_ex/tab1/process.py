import re
import numpy as np

# Raw GIN result string
raw_data_gin = """
Cora_Non_Edge_40.00_ArScore_NoneGAT_mlp_score_inter0.10_intra0.50_total40_,7.81 ± 3.57,33.65 ± 3.90,99.38 ± 0.15,99.22 ± 0.17
Cora_Non_Edge_200.00_ArScore_NoneGAT_mlp_score_inter0.10_intra0.50_total200_,13.64 ± 9.83,30.05 ± 4.03,99.05 ± 0.07,98.86 ± 0.10
Cora_Non_Edge_800.00_ArScore_NoneGAT_mlp_score_inter0.10_intra0.50_total800_,10.52 ± 8.25,23.15 ± 8.39,97.88 ± 0.17,97.52 ± 0.33
Cora_Non_Edge_1400.00_ArScore_NoneGAT_mlp_score_inter0.10_intra0.50_total1400_,15.51 ± 5.88,30.47 ± 2.12,97.48 ± 0.22,97.57 ± 0.21
Cora_Non_Edge_2400.00_ArScore_NoneGAT_mlp_score_inter0.10_intra0.50_total2400_,11.65 ± 1.47,18.91 ± 1.15,95.85 ± 0.21,95.85 ± 0.18
Cora_Non_Edge_3600.00_ArScore_NoneGAT_mlp_score_inter0.10_intra0.50_total3600_,14.41 ± 7.89,22.70 ± 5.51,95.43 ± 0.24,95.40 ± 0.08
Cora_Non_Edge_4000.00_ArScore_NoneGAT_mlp_score_inter0.10_intra0.50_total4000_,12.32 ± 3.64,23.35 ± 3.56,95.24 ± 0.49,95.35 ± 0.47
Cora_Non_Edge_5600.00_ArScore_NoneGAT_mlp_score_inter0.10_intra0.50_total5600_,10.72 ± 5.89,15.40 ± 4.91,92.89 ± 0.17,92.38 ± 0.59

"""

# Define metric names and reversed x-axis (0.0 to 1.0 for plotting)
metric_names = ["Hits@1", "MRR", "AUC", "AP"]
x_vals = np.linspace(0.0, 1.0, 9).round(4).tolist()  # 8 values increasing

# Extract mean ± std values using regex
pattern = r'([\d.]+) ± ([\d.]+)'
matches = re.findall(pattern, raw_data_gin)

# Convert to numpy arrays and reshape to (8 rows, 4 metrics)
means = np.array([float(m[0]) for m in matches]).reshape(-1, 4)
stds = np.array([float(m[1]) for m in matches]).reshape(-1, 4)

# Reverse the rows (because input is from large to small, but x is from 0 to 1)
means = means[::-1]
stds = stds[::-1]

# Format into the desired dictionary structure
formatted_dict = {
    "GCN": {
        metric: (
            x_vals,
            means[:, i].tolist(),
            stds[:, i].tolist()
        )
        for i, metric in enumerate(metric_names)
    }
}

# Optional: pretty print
from pprint import pprint
pprint(formatted_dict)
