import pandas as pd

# print("OLD DATABASE")
# df = pd.read_csv("input.csv")
# col = df["Inh Power"]
# std = col.std()
# mean = col.mean()
# print("Std: ", var)
# print("Mean: ", mean)
# print("Num: ", col.size)
# print("Max: ", col.max())
# print("Min: ", col.min())

print("NEW DATABASE")
df = pd.read_csv("SMILES12.csv")
col = df["Inh Power"]
print("Std: ", col.std())
print("Mean: ", col.mean())
print("Num: ", col.size)
print("Max: ", col.max())
print("Min: ", col.min())
print("Range: ", col.max() - col.min())