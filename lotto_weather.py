# Import necessary libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Manually set the column names
columns = ['temp', 'pressure', 'humidity', 'slot', 'partofday', 'target']

# Load your dataset (replace 'your_dataset.csv' with your actual file)
data = pd.read_csv('3D-weather.csv', names=columns, header=0)

# Assuming 'target' is the column you want to predict
X = data[['temp', 'pressure', 'humidity', 'slot', 'partofday']]
y = data['target']

# Create a Decision Tree classifier
classifier = DecisionTreeClassifier()

# Train the classifier
classifier.fit(X.values, y)

# Manually input features for prediction
temperature = float(input("Enter temperature: "))
pressure = float(input("Enter pressure: "))
humidity = float(input("Enter humidity: "))
partofday = float(input("Enter part of day 1=morning, 2=afternoon, 3=evening: "))
slot = float(input("Enter slot:"))

# Make a prediction
prediction = classifier.predict([[temperature, pressure, humidity, slot, partofday]])

# Display the predicted classification
print(f'The predicted classification is: {prediction[0]}')