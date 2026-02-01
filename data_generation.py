import numpy as np
import pandas as pd

def generate_data(n_steps=2000):
    time = np.arange(n_steps)

    trend = time * 0.005
    seasonal = np.sin(2 * np.pi * time / 24)

    series1 = trend + seasonal + np.random.normal(0, 0.2, n_steps)
    series2 = 0.5 * series1 + np.random.normal(0, 0.1, n_steps)
    series3 = -0.3 * series1 + np.random.normal(0, 0.1, n_steps)

    df = pd.DataFrame({
        "var1": series1,
        "var2": series2,
        "var3": series3
    })

    return df

if __name__ == "__main__":
    df = generate_data()
    df.to_csv("multivariate_series.csv", index=False)
