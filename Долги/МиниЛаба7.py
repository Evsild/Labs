import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

def generate_datasets():
    """Генерация 5 различных типов наборов данных для классификации"""
    n_samples = 500
    random_seed = 30
    datasets_list = [
        datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_seed),
        datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=random_seed),
        datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 0.5], random_state=random_seed, centers=2),
        datasets.make_blobs(n_samples=n_samples, random_state=170, centers=2),
        datasets.make_blobs(n_samples=n_samples, random_state=random_seed, centers=2)
    ]
    x, y = datasets_list[3]
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    datasets_list[3] = (np.dot(x, transformation), y)
    return datasets_list
def initialize_models():
    return [
        ('Метод k-ближайших соседей', KNeighborsClassifier(n_neighbors=5)),
        ('Метод опорных векторов', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)),
        ('Дерево решений', DecisionTreeClassifier(max_depth=4, random_state=42)),
        ('Случайный лес', RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)),
        ('Наивный Байес', GaussianNB())
    ]
def visualize_decision_boundaries(models, datasets):
    fig, axes = plt.subplots(len(datasets), len(models), figsize=(20, 20))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    dataset_names = ['Концентрические круги', 'Разделенные полумесяцы',
                     'Кластеры с разной дисперсией', 'Анизотропные данные',
                     'Изотропные кластеры']
    for i, (data, name) in enumerate(zip(datasets, dataset_names)):
        X, y = data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        for j, (model_name, model) in enumerate(models):
            model.fit(X_train, y_train)
            plot_decision_surface(axes[i, j], model, X, y, X_train, y_train, X_test, y_test,
                                  f"{name}\n{model_name}")
    plt.suptitle('Сравнение методов классификации на различных типах данных', fontsize=16, y=0.99)
    plt.show()
def plot_decision_surface(ax, model, X, y, X_train, y_train, X_test, y_test, title):
    h = 0.02  # Шаг сетки
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    if hasattr(model, 'predict_proba'):
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    ax.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu,
               edgecolors='k', s=40, alpha=0.7, label='Обучающая выборка')
    y_pred = model.predict(X_test)
    test_colors = ['lime' if y_test[i] == y_pred[i] else 'red' for i in range(len(y_test))]
    ax.scatter(X_test[:, 0], X_test[:, 1], c=test_colors, marker='s',
               edgecolors='k', s=60, alpha=0.9, label='Тестовая выборка')
    ax.set_title(title, fontsize=10)
    ax.set_xticks(())
    ax.set_yticks(())
if __name__ == "__main__":
    datasets = generate_datasets()
    models = initialize_models()
    visualize_decision_boundaries(models, datasets)