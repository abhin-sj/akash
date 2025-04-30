import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Load the dataset
insurance_dataset = pd.read_csv('C:/Users/nikhi/OneDrive/Desktop/Project/insurance.csv')

# Data preprocessing: Replace categorical variables with numerical values
insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)

# Separate features and target
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Train the model
model = LinearRegression()
model.fit(X_train, Y_train)

# Save the model using Pickle
with open('insurance_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Optional: You can also evaluate the model if needed
# training_data_prediction = model.predict(X_train)
# test_data_prediction = model.predict(X_test)

# Print model accuracy if you want to verify (optional)
# print('Training R^2 Score:', model.score(X_train, Y_train))
# print('Testing R^2 Score:', model.score(X_test, Y_test))