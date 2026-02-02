import torch
import torch.nn as nn
from transformer_model import EncoderDecoderTransformer

model = EncoderDecoderTransformer(input_dim=1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

best_val_loss = float("inf")
patience = 5
counter = 0

for epoch in range(60):
    model.train()
    optimizer.zero_grad()

    output = model(X_train, X_train)
    train_loss = criterion(output, y_train)
    train_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(X_val, X_val)
        val_loss = criterion(val_output, y_val)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break
