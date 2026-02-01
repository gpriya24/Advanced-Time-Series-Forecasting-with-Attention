import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, lookback=24):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

def preprocess(df, lookback=24):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = create_sequences(scaled, lookback)

    train_size = int(0.7 * len(X))
    val_size = int(0.85 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:val_size], y[train_size:val_size]
    X_test, y_test = X[val_size:], y[val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test
