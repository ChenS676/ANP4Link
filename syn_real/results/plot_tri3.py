import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your real values)
ear = np.array([0.00, 0.12, 0.25, 0.37, 0.50, 0.62, 0.75, 0.87])

gcn      = np.array([99.5, 96.5, 94.5, 90.0, 85.5, 86.0, 83.5, 79.0])
gat      = np.array([99.7, 99.5, 99.3, 98.8, 96.5, 95.8, 95.5, 93.0])
sage     = np.array([98.0, 96.8, 95.0, 88.0, 82.0, 80.0, 85.0, 78.5])
chebgcn  = np.array([99.8, 99.7, 99.6, 99.4, 97.8, 97.5, 97.0, 96.8])
linkx    = np.array([94.0, 94.8, 95.0, 94.0, 94.2, 94.5, 94.8, 93.8])
mixhop   = np.array([97.0, 96.0, 94.0, 89.5, 80.0, 0.0, 0.0, 0.0])

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

fig, ax = plt.subplots(figsize=(4, 3))  # <-- main figure size

ax.plot(ear, gcn,     marker="o", linewidth=1.8, label="GCN")
ax.plot(ear, gat,     marker="o", linewidth=1.8, label="GAT")
ax.plot(ear, sage,    marker="o", linewidth=1.8, label="SAGE")
ax.plot(ear, chebgcn, marker="o", linewidth=1.8, linestyle="--", label="ChebGCN")
ax.plot(ear, linkx,   marker="o", linewidth=1.8, linestyle="--", label="LINKX")
ax.plot(ear, mixhop,  marker="o", linewidth=1.8, label="MixHopGCN")

ax.set_xlabel("EAR")
ax.set_ylabel("AUC (/% )")

ax.set_xlim(0.0, 0.87)
ax.set_ylim(80, 100)
ax.set_xticks(ear)
ax.set_yticks(range(80, 101, 5))

ax.legend(frameon=False, loc="lower left")

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('tri3_plot.png', dpi=300)
