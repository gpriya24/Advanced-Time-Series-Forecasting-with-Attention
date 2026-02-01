import torch
from metrics import rmse, mae, mape

def evaluate_model(model, X_test, y_test):
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32)
    preds = model(X_test).detach().numpy()

    print("RMSE:", rmse(y_test, preds))
    print("MAE:", mae(y_test, preds))
    print("MAPE:", mape(y_test, preds))
