import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Read data from a CSV file
def read_data(file_path, column_index=0):
    dataframe = pd.read_csv(file_path)
    return dataframe.iloc[:, column_index].values.reshape(-1, 1)

# Specify the CSV file path
csv_file_path = '655.csv'  # Replace with the actual CSV file path
# Specify the column index (zero-based) for the series of numbers
column_index = 8  # Replace with the desired column index

# Read data from the CSV file
X = read_data(csv_file_path, column_index)

# Corresponding output values (you can modify this based on your use case)
y = np.arange(1, len(X) + 1)

# Create SVR model
svr = SVR(kernel='linear')
svr.fit(X, y)

# Predict the next number
next_number = svr.predict([[X[-1][0] + 1]])
print(f"The predicted next number is: {next_number[0]}")

# Plotting the results
plt.scatter(X, y, color='black', label='Data points')
plt.plot(X, svr.predict(X), color='blue', label='SVR Prediction')
plt.scatter([[X[-1][0] + 1]], next_number, color='red', label='Predicted Next Number')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
