import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load the data from the CSV files
breakdown_data = pd.read_csv("Breakdown.csv")
leakage_data = pd.read_csv("/content/LeakageIV.csv")
turnon_data = pd.read_csv("TurnOn.csv")

# Convert the data to numpy arrays
breakdown_values = breakdown_data.values
leakage_values = leakage_data.values
turnon_values = turnon_data.values

# Combine the data into a single numpy array
data = np.concatenate((breakdown_values, leakage_values, turnon_values))

# Scale the data using a MinMaxScaler
scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# Define the neural network architecture
model = Sequential()
model.add(Dense(64, input_dim=data.shape[1], activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the data
model.fit(data, np.zeros(data.shape[0]), epochs=10, batch_size=64)

# Use the trained model to predict the anomaly scores of new data
new_data = np.array([(2.00E+00, 5.81E-09), (2.02E+00, 5.89E-09)])
new_data = scaler.transform(new_data)
anomaly_scores = model.predict(new_data)
print(anomaly_scores)
