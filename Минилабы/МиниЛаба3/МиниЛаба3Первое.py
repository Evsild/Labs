from matplotlib import pyplot as plt
import numpy as np
a=float(input("Введите a: "))
b=float(input("Введите b: "))
x=np.linspace(0,20,400)
y=np.cos(x)**3 + a * x * np.sin(b * x)**2
plt.plot(x,y, '-b')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['x', 'y'])
plt.show()