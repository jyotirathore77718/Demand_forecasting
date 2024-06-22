import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Creating the dataframe
data_dict = {
    'month_num': list(range(1, 13)),
    'facecream': [2500, 2630, 2140, 3400, 3600, 2760, 2980, 3700, 3540, 1990, 2340, 2900],
    'facewash': [1500, 1200, 1340, 1130, 1740, 1555, 1120, 1400, 1780, 1890, 2100, 1760],
    'toothpaste': [5200, 5100, 4550, 5870, 4560, 4890, 4780, 5860, 6100, 8300, 7300, 7400],
    'bathingsoap': [9200, 6100, 9550, 8870, 7760, 7490, 8980, 9960, 8100, 10300, 13300, 14400],
    'shampoo': [1200, 2100, 3550, 1870, 1560, 1890, 1780, 2860, 2100, 2300, 2400, 1800],
    'moisturizer': [1500, 1200, 1340, 1130, 1740, 1555, 1120, 1400, 1780, 1890, 2100, 1760],
    'total_units': [21100, 18330, 22470, 22270, 20960, 20140, 29550, 36140, 23400, 26670, 41280, 30020],
    'total_profit': [211000, 183300, 224700, 222700, 209600, 201400, 295500, 361400, 234000, 266700, 412800, 300200]
}

data = pd.DataFrame(data_dict)

# Set the month_num as the index
data.set_index('month_num', inplace=True)

# Plot the total_profit data
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['total_profit'], marker='o')
plt.title('Total Profit Over Time')
plt.xlabel('Month Number')
plt.ylabel('Total Profit')
plt.grid(True)
plt.show()

# Split data into train and test sets
train = data[:int(0.8*len(data))]
test = data[int(0.8*len(data)):]

# Plot the training and testing data
plt.figure(figsize=(10, 6))
plt.plot(train['total_profit'], label='Train')
plt.plot(test['total_profit'], label='Test')
plt.legend()
plt.show()

# Define the model
model = SARIMAX(train['total_profit'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False)

# Fit the model
sarima_fit = model.fit(disp=False)

# Print the model summary
print(sarima_fit.summary())

# Forecast
n_periods = len(test)
forecast = sarima_fit.get_forecast(steps=n_periods)
forecast_index = test.index

# Get forecast values
forecast_values = forecast.predicted_mean
conf_int = forecast.conf_int()

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(train['total_profit'], label='Train')
plt.plot(test['total_profit'], label='Test')
plt.plot(forecast_index, forecast_values, label='Forecast')
plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='k', alpha=0.1)
plt.legend()
plt.show()

# Evaluate the model
mse = mean_squared_error(test['total_profit'], forecast_values)
print(f'Test MSE: {mse:.3f}')

# Extend the forecast
future_steps = 12  # For example, predict next 12 months
future_forecast = sarima_fit.get_forecast(steps=future_steps)
future_index = pd.RangeIndex(start=test.index[-1] + 1, stop=test.index[-1] + 1 + future_steps, step=1)

# Get future forecast values
future_forecast_values = future_forecast.predicted_mean
future_conf_int = future_forecast.conf_int()

# Plot the future forecast
plt.figure(figsize=(10, 6))
plt.plot(data['total_profit'], label='Observed')
plt.plot(future_index, future_forecast_values, label='Future Forecast')
plt.fill_between(future_index, future_conf_int.iloc[:, 0], future_conf_int.iloc[:, 1], color='k', alpha=0.1)
plt.legend()
plt.show()
