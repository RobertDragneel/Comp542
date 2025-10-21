import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the CSV
df = pd.read_csv("Sunspots.csv", parse_dates=['Date'], index_col='Date')
df.rename(columns={'Monthly Mean Total Sunspot Number': 'Sunspots'}, inplace=True)

# Train/test split
split_index = int(len(df) * 0.8)
train_arima = df.iloc[:split_index]
test_arima = df.iloc[split_index:]

# Fit SARIMA model
model_arima = SARIMAX(train_arima['Sunspots'], order=(3, 1, 3), seasonal_order=(1, 1, 1, 12))
model_fit = model_arima.fit(disp=False)

# Forecast
forecast_arima = model_fit.forecast(steps=len(test_arima))

# Evaluate
rmse_arima = np.sqrt(mean_squared_error(test_arima['Sunspots'], forecast_arima))
print(f"ARIMA RMSE: {rmse_arima:.2f}")

# Plot
plt.figure(figsize=(12,6))
plt.plot(test_arima.index, test_arima['Sunspots'], label='Actual')
plt.plot(test_arima.index, forecast_arima, label='ARIMA Forecast')
plt.title('Sunspot Forecast with ARIMA')
plt.xlabel('Date')
plt.ylabel('Monthly Mean Sunspot Number')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
