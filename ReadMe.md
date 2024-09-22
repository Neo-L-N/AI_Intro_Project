# **AI Introduction Project: Linear Regression Model**

Welcome to this beginner-friendly AI project! In this project, you'll use Python to build a **Linear Regression Model** that predicts house prices based on various features like the size of the house, number of bedrooms, and the age of the house.

---

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [How to Run](#how-to-run)
5. [Explanation of the Model](#explanation-of-the-model)
6. [Results](#results)
7. [Next Steps](#next-steps)
8. [🎉 Congratulations!](#congratulations)

---

## **1. Project Overview**

This project introduces Python basics for AI by guiding you through the creation of a  **Linear Regression Model** . You'll explore:

* Loading and preparing a dataset of house prices.
* Building a simple linear regression model.
* Visualizing the model's predictions vs. actual house prices.
* Evaluating the model's performance using error metrics like  **Mean Squared Error (MSE)** ,  **Mean Absolute Error (MAE)** , and  **R-squared (R²)** .

The project gives you hands-on experience with key Python libraries like `numpy`, `pandas`, `matplotlib`, and `scikit-learn`.


## **2. Project Structure**

Here is how the project is organized:

```
AI_Intro_Project/
│
├── data/
│   └── housing_data.csv        # Fake housing dataset used in this project
│
├── src/
│   ├── data_preparation.py     # Script to load and prepare the dataset
│   ├── linear_regression.py    # Script to build and evaluate the linear regression model
│   ├── visualization.py        # Script to plot actual vs predicted values
│
├── requirements.txt            # List of dependencies
└── README.md                   # This documentation

```

### **Files:**

* **`data/housing_data.csv`** : Contains the dataset with house sizes, bedrooms, age, and prices.
* **`src/data_preparation.py`** : Loads and prepares the dataset.
* **`src/linear_regression.py`** : Builds and evaluates the linear regression model.
* **`src/visualization.py`** : Plots the actual vs predicted house prices.
* **`requirements.txt`** : Lists the required Python libraries for the project.
* **`README.md`** : Project documentation.


## **3. Requirements**

Make sure you have Python installed, and use the following command to install the required libraries:

```
pip install -r requirements.txt
```

The required libraries are:

* `numpy`
* `pandas`
* `matplotlib`
* `scikit-learn`

## **4. How to Run**

Follow these steps to run the project:

1. **Navigate to the project directory** :

   ```
   cd path_to_your_project_directory/AI_Intro_Project
   ```
2. **Install the required dependencies** :

   ```
   pip install -r requirements.txt
   ```
3. **Run the Python scripts** :

   * To build and evaluate the linear regression model:

     ```
     python src/linear_regression.py
     ```
   * To visualize the actual vs predicted prices:

     ```
     python src/visualization.py
     ```



## **5. Explanation of the Model**

### **Linear Regression Overview**

Linear regression is a method for predicting a target variable (in this case, house price) based on one or more input features (like house size). The model tries to fit a line through the data points that minimizes the difference between the predicted and actual values.

### **Dataset**

The dataset contains the following columns:

* **Size (sq ft)** : The size of the house in square feet.
* **Bedrooms** : The number of bedrooms in the house.
* **Age (years)** : The age of the house in years.
* **Price (USD)** : The price of the house in USD.

In this project, we use the **Size (sq ft)** as the primary feature to predict the  **Price (USD)** . However, you can extend this model to include more features like `Bedrooms` and `Age (years)`.


## **6. Results**

After running the `linear_regression.py` script, you will see the following results:

### **Mean Squared Error (MSE)**

The **MSE** represents the average squared difference between the actual and predicted prices. In this case, the MSE is  **13,474,444.17** , which might seem large because house prices are in the hundreds of thousands.

### **Residuals (Actual - Predicted)**

The residuals show the difference between actual and predicted house prices for each data point. The **maximum residual** is around  **$7530** , meaning that the largest prediction error is about this value.

### **Mean Absolute Error (MAE)**

The **MAE** of **$2932.81** tells you that, on average, the model's predictions are off by about $2932.

### **R-squared (R²)**

The **R-squared (R²)** value is  **0.99** , meaning the model explains 99% of the variance in house prices. This is a very strong fit.

---

## **7. Next Steps**

Here are some suggestions for improving and extending the project:

### **Add More Features** :

Currently, the model only uses house size to predict house prices. You can include additional features like the number of bedrooms and the age of the house to improve the accuracy of the model.

To include more features, modify the code in `linear_regression.py` as follows:

```
X = df[['Size (sq ft)', 'Bedrooms', 'Age (years)']]
```


### **Feature Scaling** :

If you add more features, consider normalizing or standardizing them so that they are on the same scale. This can improve model performance.

### **Try Different Models** :

Experiment with other machine learning models, such as:

* Decision Trees
* Random Forests
* Neural Networks

### **Cross-Validation** :

You can implement **k-fold cross-validation** to ensure that the model performs well on different subsets of the data.

---

## **8. 🎉 Congratulations!**

Congratulations on completing this project! You have successfully built and evaluated a **Linear Regression Model** in Python, taking your first steps into the world of AI and machine learning.

You've learned how to:

* Load and prepare a dataset.
* Build and evaluate a linear regression model.
* Visualize the results.
* Use key evaluation metrics like MSE, MAE, and R².

Now that you've gained some hands-on experience, continue exploring more advanced models and datasets to improve your machine learning skills!
