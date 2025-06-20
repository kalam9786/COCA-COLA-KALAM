# 1️⃣ Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf
import streamlit as st
import warnings

warnings.filterwarnings('ignore')

# 2️⃣ Load Historical Data
data = pd.read_csv('/Users/kalamhussainshaik/Downloads/Coca-Cola_stock_history.csv')

# 3️⃣ Data Preprocessing
data['Date'] = pd.to_datetime(data['Date'], format='ISO8601', errors='coerce')
data.dropna(subset=['Date'], inplace=True)

# Ensure Dividends and Stock Splits columns exist
if 'Dividends' not in data.columns:
    data['Dividends'] = 0
if 'Stock Splits' not in data.columns:
    data['Stock Splits'] = 0

# 4️⃣ Feature Engineering
data['MA_20'] = data['Close'].rolling(window=20).mean()
data['MA_50'] = data['Close'].rolling(window=50).mean()
data['Daily_Return'] = data['Close'].pct_change()
data['Volatility'] = data['Daily_Return'].rolling(window=20).std()
data.dropna(inplace=True)

# 5️⃣ Model Preparation
features = ['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits',
            'MA_20', 'MA_50', 'Daily_Return', 'Volatility']
target = 'Close'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False, random_state=42)

# 6️⃣ Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7️⃣ Evaluate
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# 8️⃣ Get Live Data (last 5 days, 1m interval)
ticker = 'KO'
live_data = yf.download(ticker, period='5d', interval='1m')

# Add required columns if missing
if 'Dividends' not in live_data.columns:
    live_data['Dividends'] = 0
if 'Stock Splits' not in live_data.columns:
    live_data['Stock Splits'] = 0

# Compute rolling features
live_data['MA_20'] = live_data['Close'].rolling(window=20).mean()
live_data['MA_50'] = live_data['Close'].rolling(window=50).mean()
live_data['Daily_Return'] = live_data['Close'].pct_change()
live_data['Volatility'] = live_data['Daily_Return'].rolling(window=20).std()
live_data.fillna(0, inplace=True)

# Get latest row
latest_features = live_data[features].iloc[-1:].copy()
live_prediction = model.predict(latest_features)[0]

# 9️⃣ Streamlit UI
st.set_page_config(page_title="Coca-Cola Stock Prediction", layout="wide")
st.title('📈 Coca-Cola Stock Price Prediction')

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Historical Chart")
    st.line_chart(data.set_index('Date')[['Close', 'MA_20', 'MA_50']])

with col2:
    st.subheader("📌 Model Evaluation")
    st.metric("Mean Squared Error", f"{mse:.2f}")
    st.metric("Mean Absolute Error", f"{mae:.2f}")
    st.metric("📍 Predicted Close Price (Live)", f"${live_prediction:.2f}")

st.divider()
st.subheader("📉 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(data.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)