import pandas as pd
import numpy as np
from numpy import random as rd
import torch
import random
import csv
import os
from datetime import datetime
from proc.smiles_to_graph import batch_from_csv
from torch_geometric.loader import DataLoader # type: ignore
from torch_geometric.data import Data
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, root_mean_squared_error

from model_analysis import plot_predictions
from model_training import train_model_batched, train_model_batched_w_valid, train_model_batched_w_valid_early, GCNModel, MPNNModel


seed_ = 786

RESULTS_FILE = "hp_search_results.csv"
CURVES_FILE = "learning_curves.csv"

CSV_HEADER   = ["timestamp","lr","patience","struct_3d","seed","fold","val_loss_final","train_loss_final", "stopped_epoch"]
CURVE_HEADER = ["timestamp","lr","epochs","struct_3d","seed","fold","epoch","train_loss","val_loss"]

def ensure_csv(path, header):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def set_seed(seed: int) -> None:
    """
    Sets the seed of all random generators such that the results can be reproduced,
    this is to ensure consistency while learning/tweaking.

    Args:
        seed (int): The given seed, this is defined earlier in the document.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rd.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set_seed(seed_)

def create_k_folds(k: int, graphs: list[Data]) -> list[list[Data]]:
    random.shuffle(graphs)
    fold_size = len(graphs) // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k-1 else len(graphs)
        folds.append(graphs[start:end])
    # folds = [graphs[i: i+ k] for i in range(0, len(graphs), k)]
    # folds = [DataLoader(fold, batch_size=2, shuffle=True) for fold in folds]
    return folds


def average_k_folds(k: int, times: int, struc: bool, learn: float, epochs: int) -> float:
    seeds = [int(random.uniform(0,1)*1000) for i in range(times)]
    avgs = []
    for i in seeds:
        set_seed(i)
        avg = run_k_fold("MPNN", k, struc, learn, epochs)
        avgs.append(avg)
    
    return np.array(avgs).mean()

def run_k_fold(typ: str, k: int, struct_3d: bool, lr: float = 1e-3, patience: int = 50, seed: int|None = None) -> list[float]:
    
    ensure_csv(RESULTS_FILE, CSV_HEADER)
    ensure_csv(CURVES_FILE, CURVE_HEADER)

    ts = datetime.now().isoformat(timespec="seconds")

    molecules_df = pd.read_csv("input.csv")
    graphs_df = batch_from_csv(molecules_df, struct_3d)
    graphs = graphs_df["graph"].to_list()
    num_node_features = graphs[0].num_node_features
    num_edge_features = graphs[0].num_edge_features

    folds = create_k_folds(k, graphs)
    fold_val_losses = []
    losses = {}

    with open(RESULTS_FILE, "a", newline="") as rf, \
        open(CURVES_FILE, "a", newline="") as cf:
        
        rw = csv.writer(rf)
        cw = csv.writer(cf)
         
    # all_val_losses =[]
    # all_train_losses = []
        rw.writerow(["", "", "", "", ""])
        for i in range(k):
            val_graphs = folds[i]
            train_graphs = [g for j in range(k) if j!= i for g in folds[j]]

            train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=2, shuffle=False)

            if typ == "GCN":
                model = GCNModel(in_channels=num_node_features, hidden_dim=64, out_dim=1)
            else:
                model = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=64, num_layers=3, out_dim=1)
            
            model, losses_train, losses_val, stopped_ep = train_model_batched_w_valid_early(
                train_loader,
                val_loader,
                model,
                lr=lr,
                patience=patience
            )

            val_final = float(np.mean(losses_val[-10:]))
            train_final = float(np.mean(losses_train[-10:]))
            fold_val_losses.append(val_final)

            rw.writerow([ts, lr, patience, struct_3d, seed, i+1, val_final, train_final, stopped_ep])

            for ep, (tl, vl) in enumerate(zip(losses_train, losses_val)):
                cw.writerow([ts, lr, 1000, struct_3d, seed, i+1, ep+1, round(tl, 6), round(vl, 6)])

    return fold_val_losses

    #     losses[f"run_{i+1}"] = (losses_train, losses_val)
    #     val_error = sum(losses_val[-10:]) / 10
    #     all_val_losses.append(val_error)
    
    # val_losses = [losses[f"run_{i+1}"][1] for i in range(k)]
    # train_losses = [losses[f"run_{i+1}"][0] for i in range(k)]

    # max_len = max(len(l) for l in val_losses)
    # val_padded = [l + [l[-1]]*(max_len - len(l)) for l in val_losses]
    # # train_padded = [l + [l[-1]]*(max_len - len(l)) for l in train_losses]

    # val = np.array(val_padded)
    # mean_val = val.mean(axis=0)
    # std_val = val.std(axis=0)
    # num_epochs = range(len(mean_val))
    # plt.figure(figsize=(8,5))
    # plt.plot(epochs, mean_val, label="Mean Validation Loss")
    # plt.fill_between(
    #     num_epochs,
    #     mean_val - std_val,
    #     mean_val + std_val,
    #     alpha=0.3
    # )
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("5-Fold Cross Validation: Validation Loss (Mean ± Std)")
    # plt.legend()
    # plt.show()

    # fig, axes = plt.subplots(k, 1, figsize=(8,12), sharex=False)
    # for i in range(k):
    #     axes[i].plot(losses[f"run_{i+1}"][0], label="Train Loss")
    #     axes[i].plot(losses[f"run_{i+1}"][1], label="Val Loss")
    #     axes[i].set_title(f"Fold {i+1}")
    #     axes[i].set_ylabel("Loss")
    #     axes[i].legend()

    # axes[-1].set_xlabel("Epoch")
    # plt.tight_layout()
    # plt.show()

    # for i, loss in enumerate(all_val_losses):
    #     print(f"Fold {i + 1}: {loss:.4f}")
    # print(f"Mean CV validation loss {sum(all_val_losses) / len(all_val_losses):.4f}")
    # return sum(all_val_losses) / len(all_val_losses)

def run_hp_search(k: int, times: int, struct_3d: bool, lr_values: list[float], patience_vals: list[int]):

    seeds = [int(random.uniform(0, 1)*100) for _ in range(times)]

    for lr in lr_values:
        for pat in patience_vals:
            all_fold_losses = []

            for seed in seeds:
                set_seed(seed)
                fold_losses = run_k_fold("MPNN", k, struct_3d, lr, pat, seed)
                all_fold_losses.extend(fold_losses)

            arr = np.array(all_fold_losses)
            print(f"lr={lr:.0e}  patience={pat}  "
                  f"mean={arr.mean():.4f}  std={arr.std():.4f}  "
                  f"n={len(arr)}")


def train_test_split():
    molecules_df = pd.read_csv("input.csv")
    graphs_df = batch_from_csv(molecules_df, True)
    graphs = graphs_df["graph"].to_list()

    num_node_features = graphs[0].num_node_features
    num_edge_features = graphs[0].num_edge_features
    test_df = graphs_df.sample(frac=0.2)
    test_list = test_df["graph"].tolist()
    train_set = graphs_df.drop(test_df.index)
    train_list = train_set["graph"].tolist()
    train_loader = DataLoader(train_list, batch_size=4, shuffle=True)
    mpnn_model = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=16, num_layers=3, out_dim=1)
    model, losses = train_model_batched(train_loader, mpnn_model, lr=5e-3, epochs=80)
    test_loader = DataLoader(test_list, batch_size=2, shuffle=False)
    plot_predictions(test_loader, mpnn_model, "MPNN")

    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

# train_test_split()

def run_3d_model(typ: str):
    molecules_df = pd.read_csv("input.csv")
    graphs_df = batch_from_csv(molecules_df, True)
    graphs = graphs_df["graph"].to_list()

    num_node_features = graphs[0].num_node_features
    num_edge_features = graphs[0].num_edge_features

    train_loader = DataLoader(graphs, batch_size=4, shuffle=True)

    if typ == "GCN":
        gcn_model = GCNModel(in_channels=num_node_features, hidden_dim=64, out_dim=1)
        model, losses = train_model_batched(train_loader, gcn_model, lr=1e-3, epochs=300)
        plot_predictions(train_loader, gcn_model, "GCN")
    
    else:
        mpnn_model = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=64, num_layers=3, out_dim=1)
        model, losses = train_model_batched(train_loader, mpnn_model, lr=1e-3, epochs=300)
        plot_predictions(train_loader, mpnn_model, "MPNN")

    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def run_2d_model(typ: str):
    molecules_df = pd.read_csv("input.csv")
    graphs_df = batch_from_csv(molecules_df, False)
    graphs = graphs_df["graph"].to_list()

    num_node_features = graphs[0].num_node_features
    num_edge_features = graphs[0].num_edge_features

    train_loader = DataLoader(graphs, batch_size=8, shuffle=True)

    if typ == "GCN":
        gcn_model = GCNModel(in_channels=num_node_features, hidden_dim=64, out_dim=1)
        model, losses = train_model_batched(train_loader, gcn_model, lr=1e-3, epochs=300)
        plot_predictions(train_loader, gcn_model, "GCN")
    
    else:
        mpnn_model = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=64, num_layers=3, out_dim=1)
        model, losses = train_model_batched(train_loader, mpnn_model, lr=1e-3, epochs=300)
        plot_predictions(train_loader, mpnn_model, "MPNN")

    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def record_results(model, dataloader) -> tuple:
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if "edge_attr" in model.forward.__code__.co_varnames:
                preds = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
            else:
                preds = model(batch.x, batch.edge_index, batch.batch).squeeze()
            
            targets = batch.y.squeeze()

            # Handle possible shape mismatches
            if preds.dim() == 0:
                preds = preds.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    return (all_targets, all_preds)

def compute_metrics(results_dict):
    r2s = []
    rmses = []

    for run, (y_true, y_pred) in results_dict.items():
        r2 = r2_score(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)

        r2s.append(r2)
        rmses.append(rmse)

    return {
        "r2_mean": np.mean(r2s),
        "r2_std": np.std(r2s),
        "rmse_mean": np.mean(rmses),
        "rmse_std": np.std(rmses)
    }

def run_3d_geom_ablation():
    results = {
        "no_geom": {},
        "with_geom": {}
    }
    molecules_df = pd.read_csv("input.csv")
    seeds = list(rd.randint(0, 1000, size=5))
    for index, i in enumerate(seeds) :
        set_seed(i)
        graphs_df_2d = batch_from_csv(molecules_df, False)
        graphs = graphs_df_2d["graph"].to_list()

        num_node_features = graphs[0].num_node_features
        num_edge_features = graphs[0].num_edge_features

        train_loader = DataLoader(graphs, batch_size=4, shuffle=True)
        mpnn_model = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=64, num_layers=3, out_dim=1)
        trained_model, losses = train_model_batched(train_loader, mpnn_model, lr=1e-3, epochs=300)
        run_results = record_results(trained_model, train_loader)

        results["no_geom"][f"Run {index + 1}"] = run_results
        r2 = r2_score(run_results[0], run_results[1])
        rmse = root_mean_squared_error(run_results[0], run_results[1])
        print(f"For run {index} with no-geom, the performance metrics are \n")
        print(f"""R^2 = {r2:.2} \n
                RMSE = {rmse:.2}""")
        

        molecules_df = pd.read_csv("input.csv")
        graphs_df_2d = batch_from_csv(molecules_df, True)
        graphs = graphs_df_2d["graph"].to_list()

        num_node_features = graphs[0].num_node_features
        num_edge_features = graphs[0].num_edge_features

        train_loader = DataLoader(graphs, batch_size=4, shuffle=True)
        mpnn_model = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=64, num_layers=3, out_dim=1)
        trained_model, losses = train_model_batched(train_loader, mpnn_model, lr=1e-3, epochs=300)
        run_results = record_results(trained_model, train_loader)

        results["with_geom"][f"Run {index + 1}"] = run_results
        r2 = r2_score(run_results[0], run_results[1])
        rmse = root_mean_squared_error(run_results[0], run_results[1])
        print(f"For run {index} on 3D-geom, the performance metrics are \n")
        print(f"""R^2 = {r2:.2} \n
                RMSE = {rmse:.2}""")

    no_geom_stats = compute_metrics(results["no_geom"])
    with_geom_stats = compute_metrics(results["with_geom"])


    delta_r2 = with_geom_stats["r2_mean"] - no_geom_stats["r2_mean"]
    delta_rmse = no_geom_stats["rmse_mean"] - with_geom_stats["rmse_mean"]

    print(f"ΔR² = {delta_r2:.3f}")
    print(f"ΔRMSE = {delta_rmse:.3f}")


def create_poster_plot():
    molecules_df = pd.read_csv("input.csv")
    graphs_df = batch_from_csv(molecules_df, False)
    graphs = graphs_df["graph"].to_list()
    num_node_features = graphs[0].num_node_features
    num_edge_features = graphs[0].num_edge_features

    folds = create_k_folds(5, graphs)
    k_targs = []
    k_preds = []
    for i in range(5):
        val_graphs = folds[i]
        train_graphs = [g for j in range(5) if j!= i for g in folds[j]]

        train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=2, shuffle=False)

        model = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=16, num_layers=3, out_dim=1)
        
        model, losses_train, losses_val, stopped_ep = train_model_batched_w_valid_early(
            train_loader,
            val_loader,
            model,
            lr=5e-3,
            patience=35
        )

        fold_targs, fold_preds = record_results(model, val_loader)
        k_targs.extend(fold_targs)
        k_preds.extend(fold_preds)
   
    plt.figure(figsize=(10, 8))

    k_targs = np.array(k_targs)
    k_preds = np.array(k_preds)

    l_df = pd.read_csv("loo_results_mpnn.csv")
    l_targs = l_df["target"].to_numpy()
    l_preds = l_df["prediction"].to_numpy()

    m_l, b_l = np.polyfit(l_targs, l_preds, 1)
    x_l = np.linspace(min(l_targs), max(l_targs), 100)
    plt.plot(x_l, m_l*x_l + b_l, label=f"Fit y = {m_l:.2f}x + {b_l:.2f} (leave-one-out)" )
    m, b = np.polyfit(k_targs, k_preds, 1)
    x = np.linspace(min(k_targs), max(k_targs), 100)
    plt.plot(x, m*x + b, label=f"Fit y = {m:.2f}x + {b:.2f} (5-fold cross-validation)")
    
    plt.plot([min(k_targs), max(k_targs)],
             [min(k_targs), max(k_targs)],
             "r--",linewidth=3, label="Ideal: y = x")
    plt.xlabel("True Value", fontsize=22)
    plt.ylabel("Predicted Value", fontsize=22)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig("poster_pic.png", dpi=600)
    plt.show()


def run_3d_k_fold(typ: str, k: int, coords, dist):

    molecules_df = pd.read_csv("input.csv")
    graphs_df = batch_from_csv(molecules_df, True)
    graphs = graphs_df["graph"].to_list()
    num_node_features = graphs[0].num_node_features
    num_edge_features = graphs[0].num_edge_features

    folds = create_k_folds(k, graphs)
    fold_val_losses = []
    losses = {}
         
    all_val_losses =[]
    all_train_losses = []
    for i in range(k):
        val_graphs = folds[i]
        train_graphs = [g for j in range(k) if j!= i for g in folds[j]]

        train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=2, shuffle=False)

        if typ == "GCN":
            model = GCNModel(in_channels=num_node_features, hidden_dim=32, out_dim=1)
        else:
            model = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=16, num_layers=3, out_dim=1)
        
        model, losses_train, losses_val, stopped_ep = train_model_batched_w_valid_early(
            train_loader,
            val_loader,
            model,
            lr=5e-3,
            patience=35
        )

        losses[f"run_{i+1}"] = (losses_train, losses_val)
        val_error = sum(losses_val[-10:]) / 10
        all_val_losses.append(val_error)
    
    val_losses = [losses[f"run_{i+1}"][1] for i in range(k)]
    train_losses = [losses[f"run_{i+1}"][0] for i in range(k)]

    max_len = max(len(l) for l in val_losses)
    val_padded = [l + [l[-1]]*(max_len - len(l)) for l in val_losses]
    train_padded = [l + [l[-1]]*(max_len - len(l)) for l in train_losses]

    val = np.array(val_padded)
    mean_val = val.mean(axis=0)
    std_val = val.std(axis=0)
    num_epochs = range(len(mean_val))
    plt.figure(figsize=(8,5))
    plt.plot(num_epochs, mean_val, label="Mean Validation Loss")
    plt.fill_between(
        num_epochs,
        mean_val - std_val,
        mean_val + std_val,
        alpha=0.3
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("5-Fold Cross Validation: Validation Loss (Mean ± Std)")
    plt.legend()
    plt.show()

    fig, axes = plt.subplots(k, 1, figsize=(8,12), sharex=False)
    for i in range(k):
        axes[i].plot(losses[f"run_{i+1}"][0], label="Train Loss")
        axes[i].plot(losses[f"run_{i+1}"][1], label="Val Loss")
        axes[i].set_title(f"Fold {i+1}")
        axes[i].set_ylabel("Loss")
        axes[i].legend()

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    plt.show()

    for i, loss in enumerate(all_val_losses):
        print(f"Fold {i + 1}: {loss:.4f}")
    print(f"Mean CV validation loss {sum(all_val_losses) / len(all_val_losses):.4f}")
    return sum(all_val_losses) / len(all_val_losses)


def ablation_test():
    seeds = [int(random.uniform(0,1)*1000) for i in range(5)]

    # 2D dataset (MPNN + GCN)
    molecules_df = pd.read_csv("input.csv")
    graphs_df = batch_from_csv(molecules_df, False)
    graphs = graphs_df["graph"].to_list()
    num_node_features = graphs[0].num_node_features
    num_edge_features = graphs[0].num_edge_features
    gcn_2d_errors = []
    mpnn_2d_errors = []
    mpnn_dist_errors = []
    mpnn_coord_errors = []
    mpnn_3d_errors = []
    for seed in seeds:
        set_seed(seed)
        folds = create_k_folds(5, graphs)
        for i in range(5):
            val_graphs = folds[i]
            train_graphs = [g for j in range(5) if j!= i for g in folds[j]]

            train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=2, shuffle=False)

            model_gcn = GCNModel(in_channels=num_node_features, hidden_dim=32, out_dim=1)
        
            model_gcn, losses_train, losses_val, stopped_ep = train_model_batched_w_valid_early(
                train_loader,
                val_loader,
                model_gcn,
                lr=5e-3,
                patience=35
            )

            val_error = sum(losses_val[-10:]) / 10
            gcn_2d_errors.append(val_error)

    for seed in seeds:
        set_seed(seed)
        folds = create_k_folds(5, graphs)
        for i in range(5):
            val_graphs = folds[i]
            train_graphs = [g for j in range(5) if j!= i for g in folds[j]]

            train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=2, shuffle=False)

            model_mpnn = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=15, num_layers=3, out_dim=1)
        
            model_mpnn, losses_train, losses_val, stopped_ep = train_model_batched_w_valid_early(
                train_loader,
                val_loader,
                model_mpnn,
                lr=5e-3,
                patience=35
            )

            val_error = sum(losses_val[-10:]) / 10
            mpnn_2d_errors.append(val_error)

    # Distance Only
    graphs_df = batch_from_csv(molecules_df, True, include_dist=True, include_coord=False)
    graphs = graphs_df["graph"].to_list()
    num_node_features = graphs[0].num_node_features
    num_edge_features = graphs[0].num_edge_features

    for seed in seeds:
        set_seed(seed)
        folds = create_k_folds(5, graphs)
        for i in range(5):
            val_graphs = folds[i]
            train_graphs = [g for j in range(5) if j!= i for g in folds[j]]

            train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=2, shuffle=False)

            model_mpnn = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=15, num_layers=3, out_dim=1)
        
            model_mpnn, losses_train, losses_val, stopped_ep = train_model_batched_w_valid_early(
                train_loader,
                val_loader,
                model_mpnn,
                lr=5e-3,
                patience=35
            )

            val_error = sum(losses_val[-10:]) / 10
            mpnn_dist_errors.append(val_error)


    # Coords Only
    graphs_df = batch_from_csv(molecules_df, True, include_dist=False, include_coord=True)
    graphs = graphs_df["graph"].to_list()
    num_node_features = graphs[0].num_node_features
    num_edge_features = graphs[0].num_edge_features

    for seed in seeds:
        set_seed(seed)
        folds = create_k_folds(5, graphs)
        for i in range(5):
            val_graphs = folds[i]
            train_graphs = [g for j in range(5) if j!= i for g in folds[j]]

            train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=2, shuffle=False)

            model_mpnn = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=15, num_layers=3, out_dim=1)
        
            model_mpnn, losses_train, losses_val, stopped_ep = train_model_batched_w_valid_early(
                train_loader,
                val_loader,
                model_mpnn,
                lr=5e-3,
                patience=35
            )

            val_error = sum(losses_val[-10:]) / 10
            mpnn_coord_errors.append(val_error)

    
    # Full 3D Model
    graphs_df = batch_from_csv(molecules_df, True, include_dist=True, include_coord=True)
    graphs = graphs_df["graph"].to_list()
    num_node_features = graphs[0].num_node_features
    num_edge_features = graphs[0].num_edge_features

    for seed in seeds:
        set_seed(seed)
        folds = create_k_folds(5, graphs)
        for i in range(5):
            val_graphs = folds[i]
            train_graphs = [g for j in range(5) if j!= i for g in folds[j]]

            train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_graphs, batch_size=2, shuffle=False)

            model_mpnn = MPNNModel(in_channels=num_node_features, edge_dim=num_edge_features, hidden_dim=15, num_layers=3, out_dim=1)
        
            model_mpnn, losses_train, losses_val, stopped_ep = train_model_batched_w_valid_early(
                train_loader,
                val_loader,
                model_mpnn,
                lr=5e-3,
                patience=35
            )

            val_error = sum(losses_val[-10:]) / 10
            mpnn_3d_errors.append(val_error)


    results = {
        "gcn_2d": gcn_2d_errors,
        "mpnn_2d": mpnn_2d_errors,
        "mpnn_dist": mpnn_dist_errors,
        "mpnn_coord": mpnn_coord_errors,
        "mpnn_3d": mpnn_3d_errors
    }

    df = pd. DataFrame(dict([(k, pd.Series(v)) for k, v in results.items()]))
    df.to_csv("ablation_results.csv", index = False)
    


# run_2d_k_fold("", 5)
# run_2d_model("GCN")

# average = average_k_folds(5, 10, False)
# print(f"Average CV error over 10 attempts: {average:.3f}")

# lrs = [1e-3, 5e-3, 1e-2]
# patience_nums = [20, 50]
# run_hp_search(5, 10, False, lrs, patience_nums)


