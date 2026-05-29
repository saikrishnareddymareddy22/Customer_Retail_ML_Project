# ===================== STEP 1 =====================
# Importing Required Libraries

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# ===================== STEP 2 =====================
# Load Customer Dataset

df = pd.read_csv("customer_retail csv file.csv")

print(df.head())


# ===================== STEP 3 =====================
# Handle Missing Values

df = df.dropna()


# ===================== STEP 4 =====================
# Select Required Columns

df = df[['Quantity', 'UnitPrice', 'Country']]


# ===================== STEP 5 =====================
# Encode Categorical Column using LabelEncoder

encoder = LabelEncoder()

df['Country_encoded'] = encoder.fit_transform(df['Country'])


# ===================== STEP 6 =====================
# Visualize Customer Data

plt.figure(figsize=(8,5))

plt.scatter(df['Quantity'], df['UnitPrice'])

plt.xlabel('Quantity')

plt.ylabel('UnitPrice')

plt.title('Scatter Plot of Quantity vs Unit Price')

plt.show()


# ===================== STEP 7 =====================
# Split Data into Training and Testing Sets

x = df[['Quantity', 'UnitPrice']]

y = df['Country_encoded']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# ===================== STEP 8 =====================
# Train Logistic Regression Model

print("LOGISTIC REGRESSION")

log_model = LogisticRegression()

log_model.fit(x_train, y_train)

y_pred_log = log_model.predict(x_test)

log_accuracy = accuracy_score(y_test, y_pred_log)

print("Accuracy:", log_accuracy)

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred_log))


# ===================== STEP 9 =====================
# Train Decision Tree Model

print("\nDECISION TREE")

dt_model = DecisionTreeClassifier()

dt_model.fit(x_train, y_train)

y_pred_dt = dt_model.predict(x_test)

dt_accuracy = accuracy_score(y_test, y_pred_dt)

print("Accuracy:", dt_accuracy)

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred_dt))


# ===================== STEP 10 =====================
# Train KNN Model

print("\nKNN")

knn_model = KNeighborsClassifier()

knn_model.fit(x_train, y_train)

y_pred_knn = knn_model.predict(x_test)

knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("Accuracy:", knn_accuracy)

print("Confusion Matrix:")

print(confusion_matrix(y_test, y_pred_knn))


# ===================== STEP 11 =====================
# Compare Model Performances using Graph

models = ['Logistic Regression', 'Decision Tree', 'KNN']

accuracies = [log_accuracy, dt_accuracy, knn_accuracy]

plt.figure(figsize=(8,5))

plt.bar(models, accuracies)

plt.xlabel('Models')

plt.ylabel('Accuracy')

plt.title('Comparison of Model Accuracies')

plt.ylim(0,1)

plt.show()