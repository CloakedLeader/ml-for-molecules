import torch
from torch import nn
from torch.nn import MSELoss, Linear, ReLU, Dropout, Sequential, L1Loss
from torch_geometric.loader import DataLoader # type: ignore
from torch_geometric.nn import GCNConv, global_mean_pool, MessagePassing, global_max_pool # type: ignore
from torch_geometric.utils import add_self_loops, degree # type: ignore
import torch.nn.functional as F


def train_model_batched(
        dataloader: DataLoader,
        model: nn.Module,
        lr: float = 1e-3,
        epochs: int = 300
) -> tuple[nn.Module, list]:
    """
    Trains a given model for a regression task over a given number of epochs, tracks
    the loss for each epoch and adjusts its optimisation after each batch.

    Args:
        dataloader (DataLoader): The dataloader which contains all the graphs and batches
            the data automatically for the model.
        model (nn.Module): The model that is being trained, this is kept general so the same
            code can be reused for GCN or MPNN.
        lr (float, optional): The learning rate for the optimiser. Defaults to 1e-3.
        epochs (int, optional): The number of epochs to train the data over. Defaults to 300.

    Returns:
        nn.Module: Returns the model, having been trained.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = L1Loss()
    # loss_fn = MSELoss()

    losses = []

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        for batch in dataloader:
            optimizer.zero_grad()
            if 'edge_attr' in model.forward.__code__.co_varnames:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
            else:
                out = model(batch.x, batch.edge_index, batch.batch).squeeze()
            target = batch.y.squeeze()

            assert out.shape == target.shape, f"{out.shape=} vs {target.shape=}"
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        # print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        avg_loss = total_loss / num_batches
        losses.append(float(avg_loss))
    
    return model, losses

def train_model_batched_w_valid(
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        model: nn.Module,
        lr: float = 1e-3,
        epochs: int = 300
) -> tuple[nn.Module, list, list]:

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = L1Loss()
    # loss_fn = MSELoss()

    train_losses = []
    val_losses = []

    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            if 'edge_attr' in model.forward.__code__.co_varnames:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
            else:
                out = model(batch.x, batch.edge_index, batch.batch).squeeze()
            target = batch.y.squeeze()

            assert out.shape == target.shape, f"{out.shape=} vs {target.shape=}"
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        train_losses.append(float(total_loss / num_batches))

        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                if 'edge_attr' in model.forward.__code__.co_varnames:
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
                else:
                    out = model(batch.x, batch.edge_index, batch.batch).squeeze()
                target = batch.y.squeeze()
                loss = loss_fn(out, target)

                val_loss += loss.item()
                val_batches += 1

        val_losses.append(float(val_loss / val_batches))
    
    return model, train_losses, val_losses


def train_model_batched_w_valid_early(
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        model: nn.Module,
        lr: float = 1e-3,
        max_epochs: int = 1000,
        patience: int = 50,
        min_delta: float = 1e-4,
        ema_alpha: float = 0.1
) -> tuple[nn.Module, list, list, int]:

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = L1Loss()
    # loss_fn = MSELoss()

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    ema_val_loss = None
    stopped_ep = max_epochs
    epochs_without_improvement = 0
    
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        for batch in train_dataloader:
            optimizer.zero_grad()
            if 'edge_attr' in model.forward.__code__.co_varnames:
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
            else:
                out = model(batch.x, batch.edge_index, batch.batch).squeeze()
            target = batch.y.squeeze()

            assert out.shape == target.shape, f"{out.shape=} vs {target.shape=}"
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        train_losses.append(float(total_loss / num_batches))

        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in valid_dataloader:
                if 'edge_attr' in model.forward.__code__.co_varnames:
                    out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).squeeze()
                else:
                    out = model(batch.x, batch.edge_index, batch.batch).squeeze()
                target = batch.y.squeeze()
                loss = loss_fn(out, target)

                val_loss += loss.item()
                val_batches += 1

        val_losses.append(float(val_loss / val_batches))

        if ema_val_loss is None:
            ema_val_loss = val_losses[-1]
        else:
            ema_val_loss = ema_alpha * val_losses[-1] + (1 - ema_alpha) * ema_val_loss
        
        if ema_val_loss < best_val_loss - min_delta:
            best_val_loss = ema_val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            stopped_ep = epoch + 1
            break
    
    return model, train_losses, val_losses, stopped_ep


class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim=1, dropout_rate=0.05):
        super().__init__()

        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim) 

        # Feedforward MLP for regression
        self.ffnn = Sequential(
            Linear(hidden_dim * 2, hidden_dim),
            ReLU(),
            Dropout(dropout_rate),
            Linear(hidden_dim, hidden_dim // 2),
            ReLU(),
            Dropout(dropout_rate),
            Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        x_res = x 
        x = self.conv2(x, edge_index)
        x = torch.relu(x) + x_res

        x_res = x
        x = self.conv3(x, edge_index)
        x = torch.relu(x) + x_res

        # batch = torch.zeros(x.size(0), dtype=torch.long)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        return self.ffnn(x)


class MPNNLayer(MessagePassing):
    def __init__(self, in_channels, edge_dim, hidden_dim):
        super().__init__(aggr='add')
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return self.norm(out)
    
    def message(self, x_j, edge_attr):  # type: ignore
        return self.message_mlp(torch.cat([x_j, edge_attr], dim=-1))
    
    def update(self, aggr_out, x):  # type: ignore
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))
    

class MPNNModel(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_dim, num_layers=3, out_dim=1, dropout_rate=0.2):
        super().__init__(aggr="mean")

        self.node_proj = nn.Linear(in_channels, hidden_dim)

        self.layers = nn.ModuleList([
            MPNNLayer(
                in_channels=hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
            )
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout_rate)

        self.ffnn = nn.Sequential(
            nn.Linear(hidden_dim *2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, out_dim),
        )
 
    def forward(self, x, edge_index, edge_attr, batch):
        # Project node features to hidden_dim
        x = self.node_proj(x)

        for layer in self.layers:
            x_res = x
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + x_res  # simple residual connection

        # Graph-level pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        return self.ffnn(x)