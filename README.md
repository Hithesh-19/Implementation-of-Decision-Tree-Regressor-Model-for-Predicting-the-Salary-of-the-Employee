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
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

df = pd.read_csv("Salary.csv")

X = df[['Level']]

y = df['Salary']

model = DecisionTreeRegressor(random_state=42)

model.fit(X, y)

prediction = model.predict([[6.5]])

print("Predicted Salary:", prediction[0])

plt.scatter(df['Level'], df['Salary'])

plt.plot(df['Level'], model.predict(X))

plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Decision Tree Regression")

plt.show()

plt.figure(figsize=(12,8))

plot_tree(model,
          feature_names=['Level'],
          filled=True)

plt.title("Decision Tree Regressor Tree")

plt.show()
```

## Output:
<img width="685" height="501" alt="Screenshot 2026-05-28 201108" src="https://github.com/user-attachments/assets/12b267db-a4a8-46d0-875a-063e28cf9c81" />
<img width="1023" height="666" alt="Screenshot 2026-05-28 201141" src="https://github.com/user-attachments/assets/350177df-adfd-4432-a52a-af7fdd41ba7b" />




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
