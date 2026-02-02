import torch
from metrics import rmse, mae, mape

def evaluate_transformer(model, X_test, y_test, scaler):
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_test, X_test).numpy()

    preds_inv = scaler.inverse_transform(preds)
    y_test_inv = scaler.inverse_transform(y_test)

    return {
        "RMSE": rmse(y_test_inv, preds_inv),
        "MAE": mae(y_test_inv, preds_inv),
        "MAPE": mape(y_test_inv, preds_inv)
    }

def evaluate_sarima(sarima_model, test_series):
    forecast = sarima_model.forecast(len(test_series))
    return {
        "RMSE": rmse(test_series, forecast),
        "MAE": mae(test_series, forecast),
        "MAPE": mape(test_series, forecast)
    }
