"""4. Arquitecturas neuronales, escalado y entrenamiento."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model):
    """
    Cuenta los parámetros entrenables de un modelo PyTorch.
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]

        # De:
        #   (window_size, n_features)
        # a:
        #   (n_features, window_size)
        x = x.transpose(0, 1)

        y = self.y[idx]

        return x, y


class SmallCNN1D(nn.Module):
    def __init__(
        self,
        n_features,
        n_channels=8,
        kernel_size=3,
        dropout=0.30,
    ):
        super().__init__()

        padding = kernel_size - 1

        self.conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.regressor = nn.Sequential(
            nn.Linear(n_channels, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        out = self.conv(x)

        # Quitar excedente del padding para conservar causalidad aproximada.
        out = out[:, :, :x.shape[-1]]

        out = self.relu(out)
        out = self.dropout(out)

        last_time_step = out[:, :, -1]

        y_hat = self.regressor(last_time_step)

        return y_hat.squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x

        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )

        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = None

        self.final_relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        return self.final_relu(out + residual)


class SmallTCN(nn.Module):
    def __init__(
        self,
        n_features,
        channels,
        kernel_size=2,
        dropout=0.30,
    ):
        super().__init__()

        layers = []

        for i, out_channels in enumerate(channels):
            dilation = 2 ** i

            if i == 0:
                in_channels = n_features
            else:
                in_channels = channels[i - 1]

            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.regressor = nn.Linear(channels[-1], 1)

    def forward(self, x):
        out = self.network(x)
        last_time_step = out[:, :, -1]
        y_hat = self.regressor(last_time_step)

        return y_hat.squeeze(-1)


def scale_fold(X_train, X_val, y_train):
    n_train, window_size, n_features = X_train.shape

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_features)
    X_val_2d = X_val.reshape(-1, n_features)

    X_train_scaled = scaler_X.fit_transform(X_train_2d)
    X_val_scaled = scaler_X.transform(X_val_2d)

    X_train_scaled = X_train_scaled.reshape(n_train, window_size, n_features)
    X_val_scaled = X_val_scaled.reshape(1, window_size, n_features)

    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    return X_train_scaled, X_val_scaled, y_train_scaled, scaler_y


def train_neural_model_for_fold(
    model,
    X_train,
    y_train,
    X_val,
    device,
    batch_size,
    epochs,
    patience,
    learning_rate,
    weight_decay,
):
    train_dataset = TimeSeriesDataset(X_train, y_train)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    model = model.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    # (1, window_size, n_features) -> (1, n_features, window_size)
    X_val_tensor = X_val_tensor.transpose(1, 2).to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            pred = model(batch_X)
            loss = criterion(pred, batch_y)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses))

        if train_loss < best_loss:
            best_loss = train_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():
        pred_scaled = model(X_val_tensor).cpu().numpy()[0]

    return float(pred_scaled)
