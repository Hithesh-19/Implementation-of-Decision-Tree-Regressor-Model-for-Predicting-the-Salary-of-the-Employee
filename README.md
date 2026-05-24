# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Initialize Dataset: Construct a dictionary containing professional positions, numerical career levels, and corresponding salaries, and load it into a pandas DataFrame.
Extract Features and Targets: Isolate the independent feature matrix ($X$) from the Level column and the dependent continuous target vector ($y$) from the Salary column.
Train Regressor Model: Instantiate a DecisionTreeRegressor with a fixed seed for reproducible splits, and train the model using the feature-target pairs.Predict Continuous Outcomes: Generate prediction values across the original data and feed an arbitrary intermediate feature value (e.g., level 6.5) into the model to predict its expected step-wise salary.
Plot Step Function Regression: Generate a highly dense, sequential array of level values (X_grid), predict their corresponding salaries, and plot the resulting continuous step function alongside the actual data points using matplotlib.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: HITHESH RAJ R K
RegisterNumber: 212225040129
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# ------------------------------
# Step 1: Sample dataset
# ------------------------------
data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant',
                 'Manager', 'Country Manager', 'Region Manager',
                 'Partner', 'Senior Partner', 'C-level', 'CEO'],
    'Level': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}

df = pd.DataFrame(data)

# ------------------------------
# Step 2: Split features and target
# ------------------------------
X = df[['Level']]     # Feature (Level)
y = df['Salary']      # Target (Salary)

# ------------------------------
# Step 3: Create Decision Tree Regressor
# ------------------------------
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X, y)

# ------------------------------
# Step 4: Predict salary for the dataset or new levels
# ------------------------------
y_pred = regressor.predict(X)
print("Predicted salaries:", y_pred)

# Example: predict salary for a new employee at level 6.5
level = np.array([[6.5]])
predicted_salary = regressor.predict(level)
print(f"Predicted Salary for level {level[0][0]}: {predicted_salary[0]}")

# ------------------------------
# Step 5: Visualize the results (High-resolution curve)
# ------------------------------
X_grid = np.arange(min(X.values), max(X.values)+0.01, 0.01)  # High-resolution for smoother curve
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red', label='Actual Salary')
plt.plot(X_grid, regressor.predict(X_grid), color='blue', label='Decision Tree Prediction')
plt.title('Decision Tree Regression: Level vs Salary')
plt.xlabel('Level')
plt.ylabel('Salary')
plt.legend()
plt.show()
```

## Output:
<img width="1301" height="782" alt="image" src="https://github.com/user-attachments/assets/7457e3ef-cb9d-480f-816e-c9fbc2454271" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
