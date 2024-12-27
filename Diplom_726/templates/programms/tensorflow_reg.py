# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (mean_squared_error, root_mean_squared_error,
                             mean_absolute_error)
import requests
import io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# для корректной работы кода на персональном комьютере
# необходимо включение параметра "os.environ", так что
# его включение происходит до подключения "tensorflow.keras"
# поэтому вынужденное нарушение PEP8
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout

# Загрузка данных
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(io.StringIO(requests.get(url).text))

# Просмотр первых строк данных
print(data.head())

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

# Изменение формы данных для LSTM [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

type_model = 5  # 1, 2, 3, 4, 5

if type_model == 1:
    model = models.Sequential([
        Input(shape=(time_step, )),
        Dense(4000, activation='linear'),
        Dense(10, activation='linear'),
        Dense(1950, activation='linear'),
        Dropout(0.2),
        Dense(7, activation='linear'),
        Dense(1050, activation='linear'),
        Dense(7, activation='linear'),
        Dense(500, activation='linear'),
        Dropout(0.2),
        Dense(6, activation='linear'),
        Dense(300, activation='linear'),
        Dense(6, activation='linear'),
        Dense(100, activation='linear'),
        Dense(25),
        Dense(1),
    ])
    batch_size_ = 3  # 3
    validation_split_ = 0.2  # 0.2
    epochs_ = 100

elif type_model == 2:
    model = models.Sequential([
        Input(shape=(time_step, )),
        Dense(4000, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1950, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='relu'),
        Dense(1050, activation='relu'),
        Dense(7, activation='relu'),
        Dense(500, activation='relu'),
        Dropout(0.2),
        Dense(6, activation='relu'),
        Dense(300, activation='relu'),
        Dense(6, activation='relu'),
        Dense(100, activation='relu'),
        Dense(25),
        Dense(1),
    ])
    batch_size_ = 35  # 35
    validation_split_ = 0.2  # 0.2
    epochs_ = 100  # 80

elif type_model == 3:
    model = models.Sequential([
        Input(shape=(time_step, )),
        Dense(units=28, activation='linear'),
        Dense(units=38, activation='linear'),
        Dense(units=52, activation='linear'),
        Dense(units=45, activation='linear'),
        Dropout(0.2),
        Dense(units=45, activation='linear'),
        Dense(units=40, activation='linear'),
        Dropout(0.2),
        Dense(units=42, activation='linear'),
        Dense(units=40, activation='linear'),
        Dense(22),
        Dense(1),
    ])
    batch_size_ = 55  # 35
    validation_split_ = 0.2  # 0.2
    epochs_ = 100  # 85

elif type_model == 4:
    model = models.Sequential([
        Input(shape=(time_step, )),
        Dense(units=28, activation='relu'),
        Dense(units=38, activation='relu'),
        Dense(units=52, activation='relu'),
        Dense(units=45, activation='relu'),
        Dropout(0.2),
        Dense(units=45, activation='relu'),
        Dense(units=40, activation='relu'),
        Dropout(0.2),
        Dense(units=42, activation='relu'),
        Dense(units=40, activation='relu'),
        Dense(22),
        Dense(1),
    ])
    batch_size_ = 35  # 35
    validation_split_ = 0.1  # 0.2
    epochs_ = 100  # 100

elif type_model == 5:
    model = models.Sequential([
        Input(shape=(time_step, 1)),
        LSTM(65, return_sequences=True),
        Dropout(0.2),
        LSTM(55, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1),
    ])
    batch_size_ = 3  # 3
    validation_split_ = 0.2  # 0.2
    epochs_ = 100

model.summary()


# Компиляция и обучение модели
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['root_mean_squared_error'])
cl = model.fit(X_train, y_train,
               batch_size=batch_size_,
               validation_split=validation_split_,  # validation_data=(X_test, y_test),  #
               epochs=epochs_)

fig, ax = plt.subplots(figsize=(8, 3))

plt.plot(cl.history['root_mean_squared_error'], label='root_mean_squared_error')
plt.plot(cl.history['val_root_mean_squared_error'], label='val_root_mean_squared_error', linestyle='--')
plt.plot(cl.history['loss'], label='loss')
plt.plot(cl.history['val_loss'], label='val_loss', linestyle='--')
plt.xlim(10, 120)
plt.ylim(0, 0.15)
plt.legend()
plt.show()

# Предсказание и оценка модели
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Обратное преобразование предсказанных значений к исходной шкале
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# дополнительные переменные с преобразованием для оценки аккуртаности
"""
y_normalized = np.transpose(scaler.inverse_transform(data_normalized))[0]
array_of_slice_start = len(y_normalized) - len(y_test[0]) - 0  # определяем границу среза данных
array_of_slice_stop = len(y_normalized) - 0
y_slice_normal = y_normalized[array_of_slice_start:array_of_slice_stop]  # срез нормализованных данных
y_test_predict = np.transpose(test_predict).astype(np.float64).round()[0]  # тестовые значения
"""
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
