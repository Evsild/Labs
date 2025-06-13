import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np

df = pd.read_excel('датасет-1.xlsx')
print(df.head())
print(df.dtypes)

if df['price'].dtype == object:
    df['price'] = df['price'].str.replace(',', '.').astype(float)

plt.scatter(df.area, df.price, color='red')
plt.xlabel('Площадь')
plt.ylabel('Стоимость')
plt.title('Зависимость стоимости от площади')
plt.show()
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

print(f"Цена квартиры 38 м2: {reg.predict([[38]])[0]:.2f} млн руб")
print(f"Цена квартиры 200 м2: {reg.predict([[200]])[0]:.2f} млн руб")
print(f"Коэффициент a: {reg.coef_[0]:.4f}")
print(f"Коэффициент b (intercept): {reg.intercept_:.4f}")
print(f"Уравнение модели: price = {reg.coef_[0]:.4f} * area + {reg.intercept_:.4f}")

plt.scatter(df.area, df.price, color='red')
plt.plot(df.area, reg.predict(df[['area']]), color='blue')
plt.xlabel('Площадь')
plt.ylabel('Стоимость')
plt.title('Линейная регрессия: стоимость vs площадь')
plt.show()

pred = pd.read_excel('prediction_price.xlsx')
if 'price' in pred.columns and pred['price'].dtype == object:
    pred['price'] = pred['price'].str.replace(',', '.').astype(float)
pred['predicted_prices'] = reg.predict(pred[['area']])
pred.to_excel('new_predictions.xlsx', index=False)
print("\nРезультаты предсказания:")
print(pred)
print("\nФайл 'new_predictions.xlsx' успешно сохранен")