from matplotlib import pyplot as plt
import numpy as np
file=open('data.txt','r').readlines()
x=[]
y=[]
for i in file:
    xk, yk = i.strip().split()
    x.append(float(xk))
    y.append(float(yk))
plt.plot(x,y, '-b')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['x', 'y'])
plt.show()