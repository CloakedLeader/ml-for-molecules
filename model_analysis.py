import pandas as pd
import numpy as np
import random
import torch
from proc.smiles_to_graph import batch_from_csv
from torch import nn
from torch.nn import MSELoss, Linear, ReLU, Dropout, Sequential
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# import plotly.express as px
from scipy.stats import pearsonr, spearmanr


def plot_predictions(dataloader: DataLoader, model: nn.Module):
    """
    Function to plot predictions vs actual values.
    """
    all_preds = [] 
    all_targets = []

    # Put model in evaluation mode
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            preds = model(batch.x, batch.edge_index, batch.batch).squeeze()
            targets = batch.y.squeeze()

            # Handle possible shape mismatches
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Convert to numpy arrays
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Compute metrics
    r2 = r2_score(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    pearson_corr, _ = pearsonr(all_targets, all_preds)
    spearman_corr, _ = spearmanr(all_targets, all_preds)

    print(f"R² score: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE: {mae:.3f}")
    print(f"Pearson r: {pearson_corr:.3f}")
    print(f"Spearman ρ: {spearman_corr:.3f}")

    errrors = np.abs(all_preds - all_targets)

    # Plot
    plt.figure(figsize=(7, 7))

    scatter = plt.scatter(all_targets, all_preds, c=errrors, cmap="viridis", alpha=0.8)
    plt.colorbar(scatter, label="Absolute Error")
    plt.plot([all_targets.min(), all_targets.max()],
             [all_targets.min(), all_targets.max()], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs. Actual - GCN Model (Batched)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("fig.png")


# def plot_predictions_plotly(dataloader, model, metadata_df=None):
#     """
#     Function to plot predictions vs actual values using Plotly with hover tooltips.
    
#     metadata_df: Optional DataFrame with molecule metadata.
#                  Should have at least as many rows as total predictions.
#                  Example columns: MoleculeName, SMILES, CAS.
#     """
#     all_preds = []
#     all_targets = []

#     # Put model in eval mode
#     model.eval()
#     with torch.no_grad():
#         for batch in dataloader:
#             preds = model(batch.x, batch.edge_index, batch.batch).squeeze()
#             targets = batch.y.squeeze()

#             if preds.dim() == 0:
#                 preds = preds.unsqueeze(0)
#             if targets.dim() == 0:
#                 targets = targets.unsqueeze(0)

#             all_preds.append(preds.cpu())
#             all_targets.append(targets.cpu())

#     # Flatten
#     all_preds = torch.cat(all_preds, dim=0).numpy()
#     all_targets = torch.cat(all_targets, dim=0).numpy()

#     # Compute metrics
#     r2 = r2_score(all_targets, all_preds)
#     rmse = np.sqrt(mean_squared_error(all_targets, all_preds))

#     print(f"R² score: {r2:.3f}")
#     print(f"RMSE: {rmse:.3f}")

#     # Build DataFrame
#     df = pd.DataFrame({
#         "Actual": all_targets,
#         "Predicted": all_preds
#     })

#     # Merge metadata if available
#     if metadata_df is not None:
#         metadata_df = metadata_df.reset_index(drop=True)
#         df = pd.concat([df, metadata_df.iloc[:len(df)].reset_index(drop=True)], axis=1)

#     # Make plot
#     fig = px.scatter(
#         df,
#         x="Actual",
#         y="Predicted",
#         hover_data={
#             "Inhibitor Name": True,
#             "Inh Power": False, #':.2f',
#             "CAS Number": False,
#             "SMILES": False
#         },
#         title="Predicted vs Actual - GCNModel (Batched)",
#         labels={"Actual": "Actual", "Predicted": "Predicted"},
#         height=600,
#         width=600
#     )

#     # Identity line y=x
#     fig.add_shape(
#         type="line",
#         x0=df["Actual"].min(),
#         y0=df["Actual"].min(),
#         x1=df["Actual"].max(),
#         y1=df["Actual"].max(),
#         line=dict(color="red", dash="dash"),
#     )

#     fig.update_traces(marker=dict(size=8, opacity=0.7))
#     fig.update_layout(showlegend=False)
#     fig.show()