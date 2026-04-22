import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # type: ignore
# import plotly.express as px

model = "GCN"

if model == "GCN":
    df = pd.read_csv("loo_results_gcn.csv")
else:
    df = pd.read_csv("loo_results.csv")

pred = df["prediction"].to_numpy()
targ = df["target"].to_numpy()
if model == "GCN":
    print("Statistical measures for leave-one-out testing on GCN Model are: \n")
else:
    print("Statisical measures for leave-one-out testing on MPNN model are: \n")
r2 = r2_score(targ, pred)
rmse = np.sqrt(mean_squared_error(targ, pred))
mae = mean_absolute_error(targ, pred)

print(f"R² score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")

plt.plot(targ, pred, "x")
plt.plot(targ, targ)
if model == "GCN":
    plt.title("Leave One Out Results for GCN Model")
    plt.savefig("GCN_LOO.png")
else:
    plt.title("Leave One Out Results for MPNN Model")
    plt.savefig("MPNN_LOO.png")
plt.show()
