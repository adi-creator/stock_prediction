import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from stock_data_fetcher import StockDataFetcher


class StockPredictor:
    def __init__(self, symbol):
        self.symbol = symbol
        self.data_fetcher = StockDataFetcher(symbol)
        self.model = LinearRegression()
        self.scaler = StandardScaler()

    def preprocess_data(self):
        data = self.data_fetcher._get_data()
        data = data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]  # Using all relevant features
        data.ffill(inplace=True)  # Forward fill missing values

        # Adding more features for better predictions
        data['Open-Close'] = data['Open'] - data['Close']
        data['High-Low'] = data['High'] - data['Low']
        data['7-Day MA'] = data['Close'].rolling(window=7).mean()
        data['14-Day MA'] = data['Close'].rolling(window=14).mean()
        data['21-Day MA'] = data['Close'].rolling(window=21).mean()

        data.dropna(inplace=True)

        data['Target'] = data['Close'].shift(-7)  # Target is the close price one week later
        data.dropna(inplace=True)
        return data

    def train_model(self, data):
        X = data[
            ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Open-Close', 'High-Low', '7-Day MA', '14-Day MA',
             '21-Day MA']]
        y = data['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardizing the features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        # Define the model and the parameters for GridSearchCV
        model = LinearRegression()
        parameters = {'fit_intercept': [True, False], 'copy_X': [True, False]}
        grid_search = GridSearchCV(model, parameters, cv=5, scoring='neg_mean_squared_error')

        grid_search.fit(X_train, y_train)

        print(f"Best Parameters: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_

        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f"Model Mean Squared Error: {mse}")

    def predict_future(self, recent_data):
        recent_dates = recent_data.index  # Extract dates before scaling
        recent_data_scaled = self.scaler.transform(recent_data)
        predictions = self.model.predict(recent_data_scaled)
        future_dates = [recent_dates[-1] + timedelta(days=i) for i in range(1, 8)]
        prediction_series = pd.Series(predictions, index=future_dates)
        return prediction_series

    def plot_results(self, data, future_predictions):
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label='Historical Close Prices')
        plt.plot(future_predictions.index, future_predictions, label='Predicted Close Prices', color='red')

        # Highlight and annotate the final predicted price
        final_predicted_date = future_predictions.index[-1]
        final_predicted_price = future_predictions.values[-1]
        plt.annotate(f'{final_predicted_price:.2f}', xy=(final_predicted_date, final_predicted_price),
                     xytext=(final_predicted_date, final_predicted_price + 5))

        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(f'Stock Price Prediction for {self.symbol}')
        plt.legend()
        plt.show()

    def run(self):
        data = self.preprocess_data()
        self.train_model(data)
        recent_data = data[-7:]  # Use the last 7 days of data for prediction
        future_predictions = self.predict_future(recent_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
                                                              'Open-Close', 'High-Low', '7-Day MA', '14-Day MA',
                                                              '21-Day MA']])
        print("Future Predictions for the next week:")
        print(future_predictions)
        self.plot_results(data, future_predictions)
