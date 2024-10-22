import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load the California Housing dataset
california_data = fetch_california_housing()
df = pd.DataFrame(california_data.data, columns=california_data.feature_names)
df['MedHouseVal'] = california_data.target

# Split the data
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Streamlit App Title
st.title('California Housing Price Prediction')

# Introduction
st.write("""
This app predicts the **Median House Price** in California using the Linear Regression model.
Please adjust the input sliders to update the predicted house price.
""")

# Create input sliders for features
MedInc = st.slider('Median Income (scaled)', min_value=float(X['MedInc'].min()), max_value=float(X['MedInc'].max()), value=float(X['MedInc'].mean()))
HouseAge = st.slider('House Age', min_value=float(X['HouseAge'].min()), max_value=float(X['HouseAge'].max()), value=float(X['HouseAge'].mean()))
AveRooms = st.slider('Average Number of Rooms', min_value=float(X['AveRooms'].min()), max_value=float(X['AveRooms'].max()), value=float(X['AveRooms'].mean()))
AveBedrms = st.slider('Average Number of Bedrooms', min_value=float(X['AveBedrms'].min()), max_value=float(X['AveBedrms'].max()), value=float(X['AveBedrms'].mean()))
Population = st.slider('Population', min_value=float(X['Population'].min()), max_value=float(X['Population'].max()), value=float(X['Population'].mean()))
AveOccup = st.slider('Average Occupancy', min_value=float(X['AveOccup'].min()), max_value=float(X['AveOccup'].max()), value=float(X['AveOccup'].mean()))
Latitude = st.slider('Latitude', min_value=float(X['Latitude'].min()), max_value=float(X['Latitude'].max()), value=float(X['Latitude'].mean()))
Longitude = st.slider('Longitude', min_value=float(X['Longitude'].min()), max_value=float(X['Longitude'].max()), value=float(X['Longitude'].mean()))

# Create a prediction input array
input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])

# Predict house price
predicted_price = model.predict(input_data)

# Display predicted price
st.subheader(f'Predicted Median House Price: ${predicted_price[0] * 100000:.2f}')

# Display model performance
st.subheader('Model Performance on Test Data:')
y_pred = model.predict(X_test)
MAE = metrics.mean_absolute_error(y_test, y_pred)
MSE = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)
st.write(f'Mean Absolute Error (MAE): {MAE:.3f}')
st.write(f'Mean Squared Error (MSE): {MSE:.3f}')
st.write(f'Root Mean Squared Error (RMSE): {RMSE:.3f}')
