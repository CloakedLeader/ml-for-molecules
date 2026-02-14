import pandas as pd
import numpy as np
import random
import torch
from proc.smiles_to_graph import batch_from_csv
from torch_geometric.loader import DataLoader # type: ignore

from model_analysis import plot_predictions
from model_training import train_gcn_model_batched, GCNModel, MPNNModel


seed_ = 786
molecules_df = pd.read_csv("input.csv")
graph_list = batch_from_csv("input.csv")

num_node_features = graph_list[0].num_node_features
num_edge_features = graph_list[0].num_edge_features

# ys = np.array([data.y for data in graph_list])
# variance = np.var(ys)  # Use ddof=1 for sample variance
# print("Variance:", variance)
# ?  This dataset has a Variance = 32.06531


def set_seed(seed: int) -> None:
    """
    Sets the seed of all random generators such that the results can be reproduced,
    this is to ensure consistency while learning/tweaking.

    Args:
        seed (int): The given seed, this is defined earlier in the document.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(seed_)


train_loader = DataLoader(graph_list, batch_size=32, shuffle=True)

gcn_model = GCNModel(in_channels=num_node_features, hidden_dim=64, out_dim=1)
mpnn_model = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=64, num_layers=3, out_dim=1)

print(mpnn_model)

train_gcn_model_batched(train_loader, mpnn_model, lr=1e-3, epochs=300)
plot_predictions(train_loader, mpnn_model)