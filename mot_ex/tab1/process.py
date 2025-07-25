import re
import numpy as np

# Raw GIN result string
raw_data_gin = """
# # MixHop
# Cora_Non_Edge_40.00_ArScore_NoneMixHopGCN_mlp_score_inter0.10_intra0.50_total40_,17.43 ± 7.03,58.61 ± 10.20,99.58 ± 0.10,99.59 ± 0.10
# Cora_Non_Edge_200.00_ArScore_NoneMixHopGCN_mlp_score_inter0.10_intra0.50_total200_,36.46 ± 14.90,60.24 ± 5.56,99.11 ± 0.17,99.23 ± 0.03
# Cora_Non_Edge_800.00_ArScore_NoneMixHopGCN_mlp_score_inter0.10_intra0.50_total800_,36.79 ± 23.05,40.19 ± 10.79,98.06 ± 0.31,98.18 ± 0.20
# Cora_Non_Edge_1400.00_ArScore_NoneMixHopGCN_mlp_score_inter0.10_intra0.50_total1400_,51.22 ± 19.80,59.75 ± 17.16,97.03 ± 0.27,97.71 ± 0.20
# Cora_Non_Edge_2400.00_ArScore_NoneMixHopGCN_mlp_score_inter0.10_intra0.50_total2400_,0.00 ± 0.00,0.12 ± 0.00,50.00 ± 0.00,50.00 ± 0.00
# Cora_Non_Edge_3600.00_ArScore_NoneMixHopGCN_mlp_score_inter0.10_intra0.50_total3600_,0.00 ± 0.00,0.12 ± 0.00,50.00 ± 0.00,50.00 ± 0.00
# Cora_Non_Edge_4000.00_ArScore_NoneMixHopGCN_mlp_score_inter0.10_intra0.50_total2400_,0.00 ± 0.00,0.12 ± 0.00,50.00 ± 0.00,50.00 ± 0.00
# Cora_Non_Edge_5000.00_ArScore_NoneMixHopGCN_mlp_score_inter0.10_intra0.50_total3600_,0.00 ± 0.00,0.12 ± 0.00,50.00 ± 0.00,50.00 ± 0.00
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
