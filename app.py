import numpy as np
import pandas as pd 
import yfinance as yf
import streamlit as st
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
model = load_model("Stock Predictions Model.h5")

# Streamlit header and user input for stock symbol
st.header('Stock Market Predictor')
stock = st.text_input('Enter Stock Symbol', 'GOOG')

# Define start and end date for historical data
start = '2012-01-01'
end = '2022-12-31'

# Function to fetch stock data with caching
@st.cache(ttl=3600)  # Cache the data for an hour
def fetch_stock_data(stock, start, end):
    try:
        return yf.download(stock, start, end)
    except Exception as e:
        return None

# Fetch stock data
data = fetch_stock_data(stock, start, end)

# # Handle data retrieval errors outside of the cached function
# if data is None:
#     st.error(f"Error fetching data for {stock} from {start} to {end}. Please check the stock symbol.")
#     st.stop()  # Stop execution if data retrieval fails

# Display the downloaded data in Streamlit
st.subheader('Stock Data')
st.write(data)

# Preprocess the data for prediction
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plot Price vs Moving Averages (MA)
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(8,6))
ax1.plot(ma_50_days, 'r', label='MA50')
ax1.plot(data.Close, 'g', label='Price')
ax1.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(8,6))
ax2.plot(ma_50_days, 'r', label='MA50')
ax2.plot(ma_100_days, 'b', label='MA100')
ax2.plot(data.Close, 'g', label='Price')
ax2.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(8,6))
ax3.plot(ma_100_days, 'r', label='MA100')
ax3.plot(ma_200_days, 'b', label='MA200')
ax3.plot(data.Close, 'g', label='Price')
ax3.legend()
st.pyplot(fig3)

# Prepare data for prediction using the trained model
x = []
y_true = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y_true.append(data_test_scale[i, 0])

x, y_true = np.array(x), np.array(y_true)

# Make predictions with the model
predictions = model.predict(x)
predictions = predictions * scaler.scale_
y_true = y_true * scaler.scale_

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4, ax4 = plt.subplots(figsize=(8,6))
ax4.plot(y_true, 'g', label='Original Price')
ax4.plot(predictions, 'r', label='Predicted Price')
ax4.set_xlabel('Time')
ax4.set_ylabel('Price')
ax4.legend()
st.pyplot(fig4)


