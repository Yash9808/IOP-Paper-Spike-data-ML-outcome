import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the combined spike amplitudes data or upload FSR data ---> B_FSR_data_ML
data = pd.read_csv('combined_spike_amplitudes.csv')

# Convert string values in 'Object' column to numerical values
label_encoder = LabelEncoder()
data['Object'] = label_encoder.fit_transform(data['Object'])

# Convert target values from string to numerical values
data['Type'] = data['Type'].map({'H': 1, 'S': 0})

# Split the data into features (X) and target (y)
X = data.drop(columns=['Type'])
y = data['Type']

# Define test sizes
test_sizes = [20, 40, 50, 70, 90]

# Initialize classifiers
rfc = RandomForestClassifier(random_state=42)
dt = DecisionTreeClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)
svc = SVC(random_state=42)  # Default SVC with RBF kernel
mlp = MLPClassifier(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)

# Initialize lists to store accuracies for each test size
rfc_accuracies = []
dt_accuracies = []
lr_accuracies = []
svc_accuracies = []
mlp_accuracies = []
knn_accuracies = []

# Iterate over test sizes
for size in test_sizes:
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size/100, random_state=42)

    # Train classifiers
    rfc.fit(X_train, y_train)
    dt.fit(X_train, y_train)
    lr.fit(X_train, y_train)
    svc.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    knn.fit(X_train, y_train)

    # Make predictions
    rfc_pred = rfc.predict(X_test)
    dt_pred = dt.predict(X_test)
    lr_pred = lr.predict(X_test)
    svc_pred = svc.predict(X_test)
    mlp_pred = mlp.predict(X_test)
    knn_pred = knn.predict(X_test)

    # Calculate accuracies
    rfc_accuracy = accuracy_score(y_test, rfc_pred)
    dt_accuracy = accuracy_score(y_test, dt_pred)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    svc_accuracy = accuracy_score(y_test, svc_pred)
    mlp_accuracy = accuracy_score(y_test, mlp_pred)
    knn_accuracy = accuracy_score(y_test, knn_pred)

    # Store accuracies
    rfc_accuracies.append(rfc_accuracy)
    dt_accuracies.append(dt_accuracy)
    lr_accuracies.append(lr_accuracy)
    svc_accuracies.append(svc_accuracy)
    mlp_accuracies.append(mlp_accuracy)
    knn_accuracies.append(knn_accuracy)



print("Random Forest Classifier Accuracies:", rfc_accuracies)
print("Decision Tree Classifier Accuracies:", dt_accuracies)
print("Logistic Regression Accuracies:", lr_accuracies)
print("MLP Classifier Accuracies:", mlp_accuracies)
print("SVC Classifier Accuracies:", svc_accuracies)
print("KNN Classifier Accuracies:", knn_accuracies)

# Plotting test size vs accuracy for each algorithm
"""plt.figure(figsize=(15, 10))

plt.plot(test_sizes, rfc_accuracies, label='Random Forest', marker='o')
plt.plot(test_sizes, dt_accuracies, label='Decision Tree', marker='o')
plt.plot(test_sizes, lr_accuracies, label='Logistic Regression', marker='o')
plt.plot(test_sizes, svc_accuracies, label='SVC (Default)', marker='o')  # Default SVC plot
plt.plot(test_sizes, mlp_accuracies, label='MLP', marker='o')
plt.plot(test_sizes, knn_accuracies, label='KNN', marker='o')

plt.title('Test Size vs Accuracy for Different Algorithms', fontsize=16)
plt.xlabel('Test Size (%)', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()"""
