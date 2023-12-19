import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def predict_next_number(file_path, column_number):
    # Read CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Extract features and target variable
    X = data.iloc[:, column_number].values.reshape(-1, 1)
    y = data.iloc[:, column_number].shift(-1).dropna().values  # Shift the target by one to predict the next number

    # Ensure the shapes of X and y are consistent
    X = X[:-1]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict the next number
    next_number = model.predict([[data.iloc[-1, column_number]]])[0]

    return next_number

# Example usage
file_path = "655.csv"
column_number = 13  # Replace 0 with the actual column number you want to use
next_number_prediction = predict_next_number(file_path, column_number)
print(f"The predicted next number is: {next_number_prediction}")
