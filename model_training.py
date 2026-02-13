import torch
from torch import nn
from torch.nn import MSELoss, Linear, ReLU, Dropout, Sequential
from torch_geometric.loader import DataLoader # type: ignore
from torch_geometric.nn import GCNConv, global_mean_pool # type: ignore


def train_gcn_model_batched(
        dataloader: DataLoader,
        model: nn.Module,
        lr: float = 1e-3,
        epochs: int = 300
) -> nn.Module:
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
    loss_fn = MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            out: torch.Tensor = model(batch.x, batch.edge_index, batch.batch).squeeze()
            target = batch.y.squeeze()

            assert out.shape == target.shape, f"{out.shape=} vs {target.shape=}"
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    
    return model


class GCNModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, out_dim=1, dropout_rate=0.2):
        super().__init__()

        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden_dim)


        # Feedforward MLP for regression
        self.ffnn = Sequential(
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Dropout(dropout_rate),
            Linear(hidden_dim, out_dim)
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        # batch = torch.zeros(x.size(0), dtype=torch.long)
        x = global_mean_pool(x, batch)
        out = self.ffnn(x)
        return out
