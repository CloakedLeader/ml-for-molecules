import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error # type: ignore
# import plotly.express as px
from scipy.stats import pearsonr, spearmanr # type: ignore

df = pd.read_csv("loo_results_gcn.csv")
pred = df["prediction"].to_numpy()
targ = df["target"].to_numpy()

r2 = r2_score(targ, pred)
rmse = np.sqrt(mean_squared_error(targ, pred))
mae = mean_absolute_error(targ, pred)
pearson_corr, _ = pearsonr(targ, pred)
spearman_corr, _ = spearmanr(targ, pred)

print(f"R² score: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"Pearson r: {pearson_corr:.3f}")
print(f"Spearman ρ: {spearman_corr:.3f}")

plt.plot(targ, pred, "x")
plt.plot(targ, targ)
plt.show()
plt.savefig("GCN_LOO.png")
