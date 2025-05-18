import matplotlib.pyplot as plt
import numpy as np
import random
from math import sqrt
from collections import Counter
xMin1=3
xMax2=20
xMax1=14
xMin2=8

yMin1=1
yMax2=10
yMax1=7
yMin2=5

pointsCount1=50
pointsCount2=50

x=[]
y=[]

for i in range(pointsCount1):
    x1=random.uniform(xMin1, xMax1)
    y1=random.uniform(yMin1, yMax1)
    x.append([x1,y1])
    y.append(0)
for i in range(pointsCount2):
    x2=random.uniform(xMin2, xMax2)
    y2=random.uniform(yMin2, yMax2)
    x.append([x2,y2])
    y.append(1)

p=0.8
def train_test_split(x, y):
    l=list(zip(x, y))
    random.shuffle(l)

    trs=int(len(l) * p)
    train=l[:trs]
    test=l[trs:]

    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)
    return x_train, y_train, x_test, y_test
x_train, y_train, x_test, y_test = train_test_split(x, y)
def eqdist(d1,d2):
    return sqrt((d1[0] - d2[0])**2 + (d1[1] - d2[1])**2)
def fit(x_train, y_train, x_test):
    k=3
    y_predict=[]
    for tep in x_test:
        distclass=[]
        for trp, tryp in zip(x_train, y_train):
            dist=eqdist(trp, tep)
            distclass.append((dist,tryp))
        distclass.sort(key=lambda x: x[0])
        knear= distclass[:k]
        knearl=[]
        for i in range(len(knear)):
            knearl.append(knear[i][1])
        mc=Counter(knearl).most_common(1)[0][0]
        y_predict.append(mc)
    return y_predict
y_predict=fit(x_train,y_train,x_test)
def accuracy(y_test, y_predict):
    KC=0
    N=len(y_test)
    for a in range(len(y_test)):
        if y_test[a]==y_predict[a]:
            KC+=1
    return KC/N
print(accuracy(y_test,y_predict))
plt.scatter(*zip(*[x_train[i] for i in range(len(x_train)) if y_train[i] == 0]), color='blue', marker='o')
plt.scatter(*zip(*[x_train[i] for i in range(len(x_train)) if y_train[i] == 1]), color='blue', marker='x')

if any(y_predict[i] == 0 and y_test[i]==1 for i in range(len(x_test))):
    plt.scatter(*zip(*[x_test[i] for i in range(len(x_test)) if (y_predict[i] == 0 and y_test[i]==1)]), color='red', marker='o')
if any(y_predict[i] == 1 and y_test[i]==0 for i in range(len(x_test))):
    plt.scatter(*zip(*[x_test[i] for i in range(len(x_test)) if (y_predict[i] == 1 and y_test[i]==0)]), color='red', marker='x')
if any(y_predict[i] == 0 and y_test[i]==0 for i in range(len(x_test))):
    plt.scatter(*zip(*[x_test[i] for i in range(len(x_test)) if (y_predict[i] == 0 and y_test[i]==0)]), color='green', marker='o')
if any(y_predict[i] == 1 and y_test[i]==1 for i in range(len(x_test))):
    plt.scatter(*zip(*[x_test[i] for i in range(len(x_test)) if (y_predict[i] == 1 and y_test[i]==1)]), color='green', marker='x')

plt.show()