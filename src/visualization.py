import matplotlib.pyplot as plt
from data_preparation import load_data
from linear_regression import perform_linear_regression

# Visualize actual vs predicted prices
def plot_predictions(df, predictions):
    plt.scatter(df['Size (sq ft)'], df['Price (USD)'], color='blue', label='Actual Prices')
    plt.scatter(df['Size (sq ft)'], predictions, color='red', label='Predicted Prices')
    plt.xlabel('Size (sq ft)')
    plt.ylabel('Price (USD)')
    plt.title('Actual vs Predicted House Prices')

    # Adjust the y-axis range to exaggerate the differences (optional)
    plt.ylim([min(df['Price (USD)']) * 0.95, max(df['Price (USD)']) * 1.05])

    plt.legend()
    plt.show()

if __name__ == "__main__":
    df = load_data()
    model, predictions = perform_linear_regression(df)
    plot_predictions(df, predictions)


