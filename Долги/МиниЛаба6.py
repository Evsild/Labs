import numpy as np
import matplotlib.pyplot as plt

def gradientDescend(func=lambda x: x ** 2, diffFunc=lambda x: 2 * x, x0=3, speed=0.01, epochs=100):
    xList = []
    yList = []
    x = x0
    for _ in range(epochs):
        x1 = x - speed * diffFunc(x)
        y = func(x1)
        xList.append(x1)
        yList.append(y)
        x = x1
    return xList, yList

func = lambda x: x ** 2 + 3 * np.sin(x)
diffFunc = lambda x: 2 * x + 3 * np.cos(x)

xList, yList = gradientDescend(func, diffFunc, x0=3, speed=0.1, epochs=100)
x_vals = np.linspace(-3, 3, 400)
y_vals = func(x_vals)
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="f(x) = x² + 3sin(x)")
plt.scatter(xList, yList, color='red', label="Градиентный спуск")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()

speedcrit = 0.4
print(f"Граничное значение speed: ~{speedcrit}")