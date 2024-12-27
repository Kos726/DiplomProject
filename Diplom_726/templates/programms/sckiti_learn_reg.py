# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_squared_error, root_mean_squared_error,
                             accuracy_score, mean_absolute_error)
import requests
import io

# Загрузка данных
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(io.StringIO(requests.get(url).text))

# Просмотр первых строк данных
# print(data.head())

# Преобразование данных
data = data['Passengers'].values.astype(float)

# Нормализация данных
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data.reshape(-1, 1))


# Создание обучающих и тестовых выборок
def create_dataset(data_, time_step_=1):
    x_, y_ = [], []
    for i_ in range(len(data_) - time_step_):
        x_.append(data_[i_:(i_ + time_step_), 0])
        y_.append(data_[i_ + time_step_, 0])
    return np.array(x_), np.array(y_)


time_step = 10
X, y = create_dataset(data_normalized, time_step)

# Разделение на обучающие и тестовые данные
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:]
y_train, y_test = y[0:train_size], y[train_size:]

# Обучение модели
model = linear_model.BayesianRidge(max_iter=5000)
model.fit(X_train, y_train)
# BayesianRidge()

# Предсказание и оценка модели
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# print(train_predict)
# print(len(train_predict))

# Обратное преобразование предсказанных значений к исходной шкале
train_predict = scaler.inverse_transform([train_predict]).round(decimals=0)[0]
test_predict = scaler.inverse_transform([test_predict]).round(decimals=0)[0]
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# дополнительные переменные с преобразованием для оценки аккуртаности
# y_normalized = np.transpose(scaler.inverse_transform(data_normalized))[0]  # Для проверки
y_normalized = data  # np.transpose(scaler.inverse_transform(data_normalized))[0]

array_of_slice_start = time_step                                            # начало среза данных
array_of_slice_node = len(y_normalized) - len(y_test[0]) - 0                # узел стыковки
array_of_slice_stop = len(y_normalized)                                     # окончание среза данных
y_slice_normal_train = y_normalized[array_of_slice_start:array_of_slice_node]   # срез нормализованных данных
y_slice_normal_test = y_normalized[array_of_slice_node:array_of_slice_stop]   # срез нормализованных данных

"""
print("time_step:", time_step)
print("len(y_test):", len(y_test[0]))

print("array_of_slice_start:", array_of_slice_start)
print("array_of_slice_node:", array_of_slice_node)
print("array_of_slice_stop:", array_of_slice_stop)
print()
print(y_normalized[array_of_slice_start:array_of_slice_node])
print(y_train[0])
print(y_slice_normal_train)
print(train_predict)
print()
print(y_normalized[array_of_slice_node:array_of_slice_stop])
print(y_test[0])
print(y_slice_normal_test)
print(len(y_test[0]))
print(test_predict)

print("Длина обучаемой кривой", train_size)
print("Длина базовой кривой", len(train_predict))
print(train_size + time_step)
print(len(test_predict))
print(train_size + time_step+len(test_predict))
"""

# Оцениваем качество модели
accuracy = accuracy_score(y_slice_normal_train, train_predict)
print(f'Accuracy train: {accuracy:.4f}')
accuracy = accuracy_score(y_slice_normal_test, test_predict)
print(f'Accuracy test: {accuracy:.4f}')

print()
mae_train = mean_absolute_error(y_slice_normal_train, train_predict)
print(f'Средняя абсолютная ошибка (Mean Absolute Error) train: {mae_train:.2f}')
mse = mean_squared_error(y_slice_normal_train, train_predict)
print(f"Среднеквадратичная ошибка по MSE train: {mse:.2f}")
rmse = root_mean_squared_error(y_slice_normal_train, train_predict)
print(f"Среднеквадратичная ошибка по RMSE train: {rmse:.2f}")

print()
mae_test = mean_absolute_error(y_slice_normal_test, test_predict)
print(f'Средняя абсолютная ошибка test (Mean Absolute Error): {mae_test:.2f}')
mse = mean_squared_error(y_slice_normal_test, test_predict)
print(f"Среднеквадратичная ошибка по MSE test: {mse:.2f}")
rmse = root_mean_squared_error(y_slice_normal_test, test_predict)
print(f"Среднеквадратичная ошибка по RMSE test: {rmse:.2f}")

# Визуализация результатов
plt.figure(figsize=(10, 6))
# Построение базовых значений
plt.plot(y_normalized, label='Исходные данные')
# Построение прогноза для обучающей выборки
# plt.plot(np.arange(time_step, train_size + time_step), y_slice_normal_train, label='Прогноз (обучение)')
plt.plot(np.arange(time_step, train_size + time_step), train_predict, label='Прогноз (обучение)')

# Построение прогноза для тестовой выборки
# plt.plot(
#    np.arange(train_size + time_step, train_size + time_step + len(test_predict)),
#    y_slice_normal_test,
#    label='Прогноз (тест)')

plt.plot(
    np.arange(train_size + time_step, train_size + time_step + len(test_predict)),
    test_predict,
    label='Прогноз (тест)')

plt.legend()
plt.xlabel('Месяцы')
plt.ylabel('Количество пассажиров')
plt.title('Прогнозирование количества международных авиапассажиров')
plt.show()
