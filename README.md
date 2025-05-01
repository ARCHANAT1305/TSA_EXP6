
## Date: 15-04-2025

# Ex.No: 6     HOLT WINTERS METHOD

### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as datetime, set it as index, and perform some initial data exploration
3. Resample it to a monthly frequency beginning of the month
4. You plot the time series data, and determine whether it has additive/multiplicative trend/seasonality
5. Split test,train data,create a model using Holt-Winters method, train with train data and Evaluate the model  predictions against test data
6. Create teh final model and predict future data and plot it

### PROGRAM:
###  NAME : ARCHANA T   
### REGISTER NUMBER :212223240013
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
data = pd.read_csv('Sunspots.csv')

# Rename columns if necessary
data.columns = [col.strip().capitalize() for col in data.columns]  # Ensure consistency
print(data.columns)  # Check the actual column names

# Convert 'Date' to datetime and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select the first 100 rows of the 'Monthly mean total sunspot number' column
sunspot_data = data[['Monthly mean total sunspot number']].head(100)



# Resample monthly (should already be monthly, but ensures consistency)
data_monthly = sunspot_data.resample('MS').sum()


# Plot original sunspot data
data_monthly.plot(title="Monthly Sunspot Counts", ylabel="Sunspots")
plt.show()

# Normalize using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly.values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

# Plot scaled data
scaled_data.plot(title="Scaled Monthly Sunspot Counts")
plt.show()

# Decompose original time series
decomposition = seasonal_decompose(data_monthly, model="additive", period=12)
decomposition.plot()
plt.show()

# Offset scaled data to avoid 0s (for multiplicative seasonality)
scaled_data += 1

# Train-test split (80/20)
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Fit Holt-Winters model
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot predictions vs test
ax = train_data.plot(label="Train")
test_predictions_add.plot(ax=ax, label="Predictions")
test_data.plot(ax=ax, label="Test")
ax.legend()
ax.set_title("Sunspot Forecast: Train vs Test")
plt.show()

# Evaluation
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("Scaled Data: Std Dev =", np.sqrt(scaled_data.var()), "Mean =", scaled_data.mean())
final_model = ExponentialSmoothing(data_monthly, trend='add', seasonal='add', seasonal_periods=12).fit()

forecast_horizon = int(len(data_monthly) / 4)
final_predictions = final_model.forecast(steps=forecast_horizon)

# Final plot
ax = data_monthly.plot(label="Observed")![image](https://github.com/user-attachments/assets/3d083557-2fd0-4753-a462-2d885e796a6c)

ax.set_title("Future Sunspot Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Sunspots")
plt.show()

```


### OUTPUT:
 ![image](https://github.com/user-attachments/assets/18705dce-4add-4e09-b703-00ec8630f373)
![image](https://github.com/user-attachments/assets/3a9220ab-b977-40ea-8dc2-2db7145c1429)

![image](https://github.com/user-attachments/assets/9634d059-09c4-46b4-8098-8a10e8b1dd55)

![image](https://github.com/user-attachments/assets/e0266f52-7437-4306-8083-c2d996232b87)

![image](https://github.com/user-attachments/assets/4ce54efd-bcff-4b01-8f4a-5fc0a5f6c555)



### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
