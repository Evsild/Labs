#Задан одномерный массив и число P. Включить элемент, равный P, после того элемента массива, который наиболее близок к среднему значению его элементов
print('Введите число P:')
p=input()
print('Введите элементы массива по отдельности. Для завершения ввода напишите "End"')
om=[]
i=''
while i!='End':
    i=input()
    if i!='End':
        om.append(int(i))
sr=sum(om)/len(om)
cmin=float('inf')
for k in range(len(om)):
    cmin=min(abs(int(om[k])-sr),cmin)
for k in range(len(om)):
    if abs(int(om[k])-sr)==cmin:
        om.insert(k+1, p)
print(om)
    
