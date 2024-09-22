import pandas as pd

# Load data from CSV
def load_data():
    df = pd.read_csv('data/housing_data.csv')
    print("Data loaded:\n", df)
    return df

if __name__ == "__main__":
    df = load_data()
