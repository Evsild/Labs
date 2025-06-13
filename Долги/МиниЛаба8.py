import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import random

def generate_data():
    def true_function(x):
        return np.sin(x) + 0.5 * x + np.log(np.abs(x) + 1)
    np.random.seed(42)
    x_values = np.linspace(-5, 5, 100).reshape(-1, 1)
    noise = np.array([random.uniform(-0.5, 0.5) for _ in range(100)]).reshape(-1, 1)
    y_values = true_function(x_values) + noise
    return x_values, y_values, true_function

def train_regression_models(X, y):
    kernel_ridge = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
    kernel_ridge.fit(X, y)
    kr_predictions = kernel_ridge.predict(X)
    kr_mse = mean_squared_error(y, kr_predictions)
    sv_regressor = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    sv_regressor.fit(X, y.ravel())
    svr_predictions = sv_regressor.predict(X)
    svr_mse = mean_squared_error(y, svr_predictions)
    forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    forest_regressor.fit(X, y.ravel())
    rf_predictions = forest_regressor.predict(X)
    rf_mse = mean_squared_error(y, rf_predictions)
    return (kernel_ridge, kr_predictions, kr_mse), (sv_regressor, svr_predictions, svr_mse), (forest_regressor, rf_predictions, rf_mse)

def plot_results(x, y, true_func, models_data):
    plt.figure(figsize=(18, 5))
    titles = [
        'Регрессия с ядром (Kernel Ridge)',
        'Метод опорных векторов (SVR)',
        'Случайный лес (Random Forest)'
    ]
    for i, (model, predictions, mse) in enumerate(models_data, 1):
        plt.subplot(1, 3, i)
        plt.scatter(x, y, color='blue', label='Зашумленные данные', s=10)
        plt.plot(x, true_func(x), color='green', label='Истинная функция')
        plt.plot(x, predictions, color='red', label='Предсказание модели')
        plt.title(f'{titles[i - 1]}\nСреднеквадратичная ошибка: {mse:.4f}')
        plt.legend()
    plt.tight_layout()
    plt.show()
def main():
    X, y, true_function = generate_data()
    kr_model, svr_model, rf_model = train_regression_models(X, y)
    plot_results(X, y, true_function, [kr_model, svr_model, rf_model])
if __name__ == "__main__":
    main()