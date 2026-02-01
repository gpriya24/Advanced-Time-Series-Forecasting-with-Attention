from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_sarima(series):
    model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,24))
    result = model.fit(disp=False)
    return result
