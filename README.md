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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("Employee.csv")

label = LabelEncoder()

df['Work_accident'] = label.fit_transform(df['Work_accident'])
df['promotion_last_5years'] = label.fit_transform(df['promotion_last_5years'])
df['Departments '] = label.fit_transform(df['Departments '])
df['salary'] = label.fit_transform(df['salary'])
df['left'] = label.fit_transform(df['left'])

X = df[['satisfaction_level',
        'last_evaluation',
        'number_project',
        'average_montly_hours',
        'time_spend_company']]

y = df['left']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(12,8))

plot_tree(model,
          feature_names=X.columns,
          class_names=['Stay', 'Left'],
          filled=True)

plt.title("Employee Churn Decision Tree")

plt.show()
```

## Output:
<img width="1040" height="742" alt="image" src="https://github.com/user-attachments/assets/f773528b-0034-44e8-a8bc-d858fd9c5a3c" />



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
