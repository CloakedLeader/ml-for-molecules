import pandas as pd

print("OLD DATABASE")
df = pd.read_csv("input.csv")
col = df["Inh Power"]
var = col.var()
mean = col.mean()
print("Variance: ", var)
print("Mean: ", mean)
print("Num: ", col.size)
print("Max: ", col.max())
print("Min: ", col.min())

# print("NEW DATABASE")
# df = pd.read_csv("SMILES12.csv")
# col = df["Inh Power"]
# print("Variance: ", col.var())
# print("Mean: ", col.mean())
# print("Num: ", col.size)