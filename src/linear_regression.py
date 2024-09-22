from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from data_preparation import load_data

# Build and evaluate the model
def perform_linear_regression(df):
    # Prepare the features and target
    X = df[['Size (sq ft)']]  # Independent variable (Size of the house)
    y = df['Price (USD)']     # Dependent variable (Price)

    # Create the Linear Regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Evaluate the model using Mean Squared Error (MSE)
    mse = mean_squared_error(y, predictions)
    print(f"Mean Squared Error: {mse:.2f}")

    return model, predictions

# Calculate and print residuals (errors)
def print_residuals(df, predictions):
    actual_prices = df['Price (USD)']
    residuals = actual_prices - predictions
    print("Residuals (Actual - Predicted):\n", residuals)
    print("Mean Absolute Error (MAE):", np.mean(np.abs(residuals)))
    print("Max Residual:", np.max(np.abs(residuals)))

# Print additional metrics: MAE and R-squared
def print_additional_metrics(y_actual, y_predicted):
    mae = mean_absolute_error(y_actual, y_predicted)
    r2 = r2_score(y_actual, y_predicted)
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

if __name__ == "__main__":
    df = load_data()
    model, predictions = perform_linear_regression(df)
    print_residuals(df, predictions)
    print_additional_metrics(df['Price (USD)'], predictions)
