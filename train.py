import torch
import torch.nn as nn
from transformer_model import TimeSeriesTransformer

def train_model(X_train, y_train, X_val, y_val, lr=0.001):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TimeSeriesTransformer(input_dim=X_train.shape[2]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)

    for epoch in range(30):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        val_loss = criterion(model(X_val), y_val)

        print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

    return model
