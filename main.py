import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional, Input
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import zscore
from keras.callbacks import EarlyStopping
from keras.losses import Huber
import datetime

# Define the stock ticker, start and end dates
ticker = "SPY"
start_date = "2020-01-01"
end_date = datetime.datetime.now().strftime("%Y-%m-%d")
# Fetch yfinance data
data = yf.download(ticker, start=start_date, end=end_date)

# Data features to use in model

close_data = data[['Close']]
volume_data = data[['Volume']]

# Calculate moving averages
data['SMA_10'] = data['Close'].rolling(window=10).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# Calculate RSI (relative strength index)
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI_14'] = calculate_rsi(data)

# Calculate daily returns
data['Daily_Return'] = data['Close'].pct_change().fillna(0)

# Fetch P/E ratio using yfinance
ticker_data = yf.Ticker(ticker)
pe_ratio = ticker_data.info.get('trailingPE', 0)
data['PE_Ratio'] = pe_ratio

data.fillna(0, inplace=True)

features = ['SMA_10', 'SMA_50', 'RSI_14', 'Daily_Return', 'PE_Ratio']
feature_data = data[features]

# MinMaxScaler to normalize data
scalers = {col: MinMaxScaler(feature_range=(0, 1)) for col in ['Close', 'Volume'] + features}
scaled_close_data = scalers['Close'].fit_transform(close_data)
scaled_volume_data = scalers['Volume'].fit_transform(volume_data)
scaled_features = [scalers[col].fit_transform(feature_data[[col]]) for col in features]

# Combine closing price, volume, and other features (7 total)
combined_data = np.concatenate([scaled_close_data, scaled_volume_data] + scaled_features, axis=1)


# Function to create training and testing datasets
def create_dataset(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i])
        y.append(data[i, 0]) # Target is closing price
    return np.array(X), np.array(y)

# Past days data using to make prediction
time_step = 90 # 3 months

# Split data to test/train
train_size = int(len(combined_data) * 0.8)
train_data = combined_data[:train_size]
test_data = combined_data[train_size - time_step:]

X_train, y_train, = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape inputs for LSTM
num_features = combined_data.shape[1]
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))


# Building LSTM model
# Bidirectional layers allow the model to have both backward and forward info about the sequence at every timestep
model = Sequential()
model.add(Input(shape=(X_train.shape[1], num_features))) 
model.add(Bidirectional(LSTM(units=100, return_sequences=True)))
model.add(Dropout(0.3))  # Dropouts to prevent overfitting
model.add(Bidirectional(LSTM(units=100, return_sequences=False)))
model.add(Dropout(0.2))
model.add(Dense(units=50))
model.add(Dense(units=1))


# Compile model using adam optimizer, Huber loss (combination of MSE and MAE), with root_mse and mae as metrics
model.compile(optimizer='adam', loss=Huber(delta=1.0), metrics=['mae', 'root_mean_squared_error'])

# Early stopping callback to prevent overfitting and stop when validation loss stops improving
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train, epochs=100, batch_size=12, validation_data=(X_test, y_test), verbose=1, callbacks=[early_stopping])


predictions = model.predict(X_test)

# Transform predictions back to original scale
predictions = scalers['Close'].inverse_transform(predictions)
y_test_unscaled = scalers['Close'].inverse_transform(y_test.reshape(-1, 1))

# Function to calculate evaluation metrics
def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return r2, mae, mse, rmse

# Evaluate LSTM model predictions
r2, mae, mse, rmse = evaluate_model(y_test_unscaled, predictions)

# Print evaluation metrics
print("LSTM Model Evaluation Metrics:")
print(f"R-squared (R2): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# Benchmark with a Naive Random Walk Model
# Random Walk assumes the next price equals the current price plus some noise
rw_predictions_unscaled = y_test_unscaled[:-1]  # Random walk predicts the previous value as the next value

# Evaluate Random Walk model predictions
r2_rw, mae_rw, mse_rw, rmse_rw = evaluate_model(y_test_unscaled[1:], rw_predictions_unscaled)

# Print evaluation metrics for Random Walk Model
print("\nNaive Model (Random Walk) Evaluation Metrics:")
print(f"R-squared (R2): {r2_rw:.4f}")
print(f"Mean Absolute Error (MAE): {mae_rw:.4f}")
print(f"Mean Squared Error (MSE): {mse_rw:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_rw:.4f}")


from sklearn.linear_model import LinearRegression

# Benchmark with a Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
linear_predictions = linear_model.predict(X_test.reshape(X_test.shape[0], -1))

# Transform linear model predictions back to original scale
linear_predictions_unscaled = scalers['Close'].inverse_transform(linear_predictions.reshape(-1, 1))

# Evaluate Linear model predictions
r2_linear, mae_linear, mse_linear, rmse_linear = evaluate_model(y_test_unscaled, linear_predictions_unscaled)

# Print evaluation metrics for Linear Regression Model
print("\nLinear Regression Model Evaluation Metrics:")
print(f"R-squared (R2): {r2_linear:.4f}")
print(f"Mean Absolute Error (MAE): {mae_linear:.4f}")
print(f"Mean Squared Error (MSE): {mse_linear:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_linear:.4f}")


pred_steps = 30 # Number of days to predict ahead (should not go over the time_step)

future_predictions = []
latest_data = X_test[-1].copy()

historical_avg_volume = np.mean(scaled_volume_data[train_size:], axis=0)[0]

# Perform predictions with a sliding window
for i in range(pred_steps):
    # Predict the next value based on the current sequence
    next_prediction = model.predict(latest_data.reshape(1, time_step, num_features))[0][0]
    next_prediction_unscaled = scalers['Close'].inverse_transform([[next_prediction]])[0][0]
    future_predictions.append(next_prediction_unscaled)

    # Update the sliding window
    actual_values = [
        y_test[-time_step + i],
        scaled_volume_data[train_size + i, 0],
        *[scalers[col].transform(np.array(feature_data.iloc[train_size + i][col]).reshape(-1, 1))[0][0] for col in features]
    ]
    latest_data = np.append(latest_data[1:], [actual_values], axis=0)
    

# Transform future predictions back to original scale
future_predictions = np.array(future_predictions).reshape(-1, 1)

# Concatenate predictions for plotting
extended_predictions = np.concatenate((predictions, future_predictions))

# Calculate price differences for anomaly detection
price_differences = []
future_steps = 14  # Number of days ahead to detect changes
for i in range(len(extended_predictions) - future_steps):
    future_average = np.mean(extended_predictions[i:i + future_steps])
    current_price = extended_predictions[i]
    price_difference = future_average - current_price
    price_differences.append(price_difference)

price_differences = np.array(price_differences).reshape(-1)

# Calculate Z-scores for price differences
z_scores = zscore(price_differences)
anomaly_threshold = 0.8  # threshold for anomaly detection

# Detect anomalies based on Z-score threshold
good_anomalies = [i for i, z in enumerate(z_scores) if z > anomaly_threshold]
bad_anomalies = [i for i, z in enumerate(z_scores) if z < -anomaly_threshold]


import matplotlib.pyplot as plt
import seaborn as sns

# Update anomalies to include all dates
all_dates = np.concatenate((data.index[train_size:], [data.index[-1] + datetime.timedelta(days=i + 1) for i in range(pred_steps)]))

# Apply Seaborn dark grid style
sns.set(style="darkgrid")
muted = sns.color_palette("muted")
muted_blue = muted[0]
muted_green = muted[2]
muted_red = muted[3]

dark = sns.color_palette("dark")
dark_blue = dark[0]
dark_green = dark[2]
dark_red = dark[3]

# Set up the figure
plt.figure(figsize=(14, 7))

plt.plot(all_dates[:len(predictions)], y_test_unscaled[:len(predictions)], color=muted_blue, label='True Price', zorder=2)
plt.axvline(x=all_dates[len(y_test_unscaled) - 1], color='gray', linestyle='--', label='Today', zorder=1)
plt.plot(all_dates[:len(predictions)], predictions, color=muted_red, label='Predicted Price (Historical)', zorder=3)
plt.plot(all_dates[len(predictions):], future_predictions, color=muted_green, label='Predicted Price (Future)', linestyle='--', zorder=3)

plt.scatter([all_dates[i] for i in good_anomalies], [extended_predictions[i] for i in good_anomalies], color=muted_green, edgecolor='green', label='Predicted Up Trend', marker='o', s=15, zorder=4)
plt.scatter([all_dates[i] for i in bad_anomalies], [extended_predictions[i] for i in bad_anomalies], color=dark_red, label='Predicted Down Trend', marker='x', s=15, zorder=4)


# Add titles and labels
plt.title(f'{ticker} Stock Price Prediction and Future Forecast')
plt.xlabel('Date')
plt.ylabel('Price')

# Add legend
plt.legend()

# Show the plot
plt.show()
