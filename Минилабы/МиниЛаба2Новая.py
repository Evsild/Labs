#Задан одномерный массив и число P. Включить элемент, равный P, после того элемента массива, который наиболее близок к среднему значению его элементов
print('Введите число P:')
p=input()
file = open("data.txt", "r")
om = []
for line in file:
    nums = list(map(float, line.strip().split()))
    om.extend(nums)
sr=sum(om)/len(om)
cmin=float('inf')
for k in range(len(om)):
    cmin=min(abs(int(om[k])-sr),cmin)
for k in range(len(om)):
    if abs(int(om[k])-sr)==cmin:
        om.insert(k+1, p)
        break
print(om)