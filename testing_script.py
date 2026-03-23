import pandas as pd
import numpy as np
from proc.smiles_to_graph import batch_from_csv
from torch_geometric.loader import DataLoader # type: ignore
import torch
from tqdm import tqdm # type: ignore

from model_training import train_gcn_model_batched, MPNNModel, GCNModel


df = pd.read_csv("input.csv")
graph_list = batch_from_csv(df)

graphs = graph_list["graph"].tolist()
num_node_features = graphs[0].num_node_features
num_edge_features =graphs[0].num_edge_features

predictions = []
targets = []

small_list = graph_list.head(3)
for idx, row in tqdm(graph_list.iterrows(), total=len(graph_list)):

    train_df = graph_list.drop(idx)
    test_df = graph_list.loc[[idx]]

    x_train = train_df["graph"].tolist()
    y_train = train_df["Inh Power"].tolist()

    x_test = test_df["graph"].iloc[0]
    y_test = test_df["Inh Power"].iloc[0]


    loader =  DataLoader(x_train, batch_size=16, shuffle=True)
    model = GCNModel(in_channels=num_node_features, hidden_dim=64, out_dim=1, dropout_rate=0.2)
    # model = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=64, num_layers=3, out_dim=1)
    train_gcn_model_batched(loader, model, lr=1e-3, epochs=300)
    model.eval()
    with torch.no_grad():
        batch_vec = torch.zeros(x_test.x.size(0), dtype=torch.long)
        if "edge_attr" in model.forward.__code__.co_varnames:
            preds = model(x_test.x, x_test.edge_index, x_test.edge_attr, batch_vec)
        else:
            preds = model(x_test.x, x_test.edge_index, batch_vec)
        pred_value = preds.squeeze().cpu().item()

    predictions.append(pred_value)
    targets.append(y_test)


results_df = pd.DataFrame({
    "prediction": predictions,
    "target": targets    
})

results_df.to_csv("loo_results_gcn.csv", index=False)
