import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

xMin1, xMax1 = 3, 14
yMin1, yMax1 = 1, 7
xMin2, xMax2 = 8, 20
yMin2, yMax2 = 5, 10
pointsCount = 50

np.random.seed(42)
class1 = np.column_stack([
    np.random.uniform(xMin1, xMax1, pointsCount),
    np.random.uniform(yMin1, yMax1, pointsCount)
])
class2 = np.column_stack([
    np.random.uniform(xMin2, xMax2, pointsCount),
    np.random.uniform(yMin2, yMax2, pointsCount)
])

X = np.vstack([class1, class2])
y = np.array([0]*pointsCount + [1]*pointsCount)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], c='blue', marker='o', label='Class 0 (train)')
plt.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], c='blue', marker='x', label='Class 1 (train)')
correct_mask = (y_pred == y_test)
plt.scatter(X_test[correct_mask & (y_test==0), 0], X_test[correct_mask & (y_test==0), 1],
            c='green', marker='o', label='Class 0 (correct)')
plt.scatter(X_test[correct_mask & (y_test==1), 0], X_test[correct_mask & (y_test==1), 1],
            c='green', marker='x', label='Class 1 (correct)')
error_mask = (y_pred != y_test)
plt.scatter(X_test[error_mask & (y_test==0), 0], X_test[error_mask & (y_test==0), 1],
            c='red', marker='o', label='Class 0 (error)')
plt.scatter(X_test[error_mask & (y_test==1), 0], X_test[error_mask & (y_test==1), 1],
            c='red', marker='x', label='Class 1 (error)')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('KNN Classification Results')
plt.legend()
plt.grid(True)
plt.show()