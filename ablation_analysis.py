import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("ablation_results.csv")

means = df.mean()

custom_labels = ["GCN (2D)", "MPNN (2D)", "MPNN (Dist)", "MPNN (Coords)", "MPNN (3D)"]
means.index = custom_labels
# means.plot(kind="bar")

fig, ax = plt.subplots()
fig.patch.set_facecolor("#e8c3af")
ax.set_facecolor("#e8c3af")

ax.bar(means.index, means.values, color="steelblue")
plt.xticks(rotation=0)
plt.xlabel("Type of Model")
plt.ylabel("Average MAE")
plt.tight_layout()
plt.savefig("ablation_bar_chart.png", dpi=600)
plt.show()