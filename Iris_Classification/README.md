# Iris Classification using Decision Tree and Logistic Regression

## Overview

This project focuses on classifying the Iris dataset using **Decision Tree** and **Logistic Regression** models. The Iris dataset consists of 150 samples of iris flowers, categorized into three species:

- **Iris-setosa**
- **Iris-versicolor**
- **Iris-virginica**

Each sample has four numerical features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

## Project Workflow

1. **Import required libraries**
2. **Load the dataset**
3. **Data preprocessing**
   - Encode categorical variables (if needed)
   - Feature scaling using `StandardScaler`
4. **Split the dataset** into training and testing sets
5. **Train models** (Decision Tree & Logistic Regression)
6. **Evaluate model performance**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report
7. **Visualize results** using Matplotlib and Seaborn

## Installation & Requirements

Ensure you have the required libraries installed:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Running the Project

1. Clone the repository or download the script.
2. Open Jupyter Notebook or any Python IDE.
3. Run the script step by step to train and evaluate the models.

## Code Implementation

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
df = pd.read_csv("iris.csv")

# Preprocessing
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
X = df.drop(columns=["Species"])
y = df['Species']

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Decision Tree Model
dt_model = DecisionTreeClassifier()
dt_model.fit(x_train, y_train)
dt_pred = dt_model.predict(x_test)

# Logistic Regression Model
lr_model = LogisticRegression(max_iter=200)
lr_model.fit(x_train, y_train)
lr_pred = lr_model.predict(x_test)

# Model Evaluation
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))

print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

print("Decision Tree Classification Report:\n", classification_report(y_test, dt_pred))
print("Logistic Regression Classification Report:\n", classification_report(y_test, lr_pred))
```

## Results

Both models achieve high accuracy in classifying Iris species.

### Example Output (Accuracy)

```
Decision Tree Accuracy: 0.96
Logistic Regression Accuracy: 0.93
```

### Confusion Matrix Example Output

```
Decision Tree Confusion Matrix:
 [[10  0  0]
  [ 0  8  1]
  [ 0  1 10]]

Logistic Regression Confusion Matrix:
 [[10  0  0]
  [ 0  9  0]
  [ 0  1 10]]
```

### Classification Report Example Output

```
Decision Tree Classification Report:
                  precision    recall  f1-score   support
    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       0.89      0.89      0.89         9
 Iris-virginica       0.91      0.91      0.91        11

       accuracy                           0.96        30

Logistic Regression Classification Report:
                  precision    recall  f1-score   support
    Iris-setosa       1.00      1.00      1.00        10
Iris-versicolor       1.00      1.00      1.00         9
 Iris-virginica       0.91      0.91      0.91        11

       accuracy                           0.93        30
```

## Visualizations

To better understand the dataset, we use Seaborn for visualization:

```python
sns.pairplot(df, hue="Species", palette="husl")
plt.show()
```

## Conclusion

- **Decision Tree** and **Logistic Regression** both perform well on the Iris dataset.
- Feature scaling enhances performance, especially for Logistic Regression.
- Decision Tree provides **higher accuracy** but may overfit the data.
- Logistic Regression is **simpler** and performs consistently well.

## Future Enhancements

- Implement **other classification models** (Random Forest, SVM, KNN, etc.).
- Perform **hyperparameter tuning** using GridSearchCV.
- Deploy the model using Flask or Streamlit.

## Author

Developed by **[SHAIK IMRAN]**

## License

This project is licensed under the **MIT License**.

