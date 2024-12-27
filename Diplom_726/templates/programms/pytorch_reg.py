import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

import requests
import io

# Загрузка данных
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
data = pd.read_csv(io.StringIO(requests.get(url).text))

# Просмотр первых строк данных
print(data.head())

data_normilize = data.iloc[:, 1:2].values


def scaling_window(data, seq_length):
    x = []
    y = []

    for i in range(len(data_normilize)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x), np.array(y)


sc = MinMaxScaler()
training_data = sc.fit_transform(data_normilize)

time_step = 10
x, y = scaling_window(training_data, time_step)

train_size = int(len(y) * 0.8)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

trainX = Variable(torch.Tensor(np.array(x[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y[0:train_size])))

testX = Variable(torch.Tensor(np.array(x[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y[train_size:len(y)])))


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_step = time_step

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        out = self.fc(h_out)

        return out


num_epochs = 2600
learning_rate = 0.001

input_size = 1
hidden_size = 5
num_layers = 1

num_classes = 1

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()

    # obtain the loss function
    loss = criterion(outputs, trainY)

    loss.backward()

    optimizer.step()
    if epoch % 100 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))


# дополнительные переменные с преобразованием для оценки качества
X_normalized = training_data  # np.transpose(scaler.inverse_transform(data_normalized))[0]

array_of_slice_start = time_step                                                        # начало среза данных
array_of_slice_node = time_step + len(trainY)                                                 # узел стыковки
array_of_slice_stop = len(X_normalized)                                     # окончание среза данных
X_slice_normal_train = X_normalized[array_of_slice_start:array_of_slice_node]   # срез нормализованных данных
X_slice_normal_test = X_normalized[array_of_slice_node:array_of_slice_stop]     # срез нормализованных данных

print("array_of_slice_node:", array_of_slice_node)
print("array_of_slice_stop:", array_of_slice_stop)

lstm.eval()
train_predict = lstm(dataX)
X_slice_predict_train = train_predict[array_of_slice_start:array_of_slice_node].detach().numpy()
X_slice_predict_test = train_predict[array_of_slice_node-11:array_of_slice_stop].detach().numpy()

print("Длина отрезка прогноза")
print(len(train_predict))
print("Длина отрезка обучения")
print(len(X_slice_normal_train))
print(len(X_slice_predict_train))
print("Длина отрезка тестироваия")
print(len(X_slice_normal_test))
print(len(X_slice_predict_test))

data_predict = train_predict.data.numpy()
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

# Оцениваем качество модели
mae_train = mean_absolute_error(X_slice_normal_train, X_slice_predict_train)
print(f'Средняя абсолютная ошибка (Mean Absolute Error) train: {mae_train:.2f}')

mse = mean_squared_error(X_slice_normal_train, X_slice_predict_train)
print(f"Среднеквадратичная ошибка по MSE train: {mse:.2f}")

rmse = root_mean_squared_error(X_slice_normal_train, X_slice_predict_train)
print(f"Среднеквадратичная ошибка по RMSE train: {rmse:.2f}")

print()
mae_test = mean_absolute_error(X_slice_normal_test, X_slice_predict_test)
print(f'Средняя абсолютная ошибка test (Mean Absolute Error): {mae_test:.2f}')
mse = mean_squared_error(X_slice_normal_test, X_slice_predict_test)
print(f"Среднеквадратичная ошибка по MSE test: {mse:.2f}")
rmse = root_mean_squared_error(X_slice_normal_test, X_slice_predict_test)
print(f"Среднеквадратичная ошибка по RMSE test: {rmse:.2f}")


# Визуализация результатов
plt.figure(figsize=(10, 6))
# Построение базовых значений


plt.plot(training_data, label='Исходные данные')
plt.plot(
    np.arange(
              array_of_slice_start,
              array_of_slice_node),
    X_slice_predict_train, label='Прогноз (обучение)')

plt.plot(
    np.arange(
              array_of_slice_node,
              array_of_slice_stop),
    X_slice_predict_test,
    label='Прогноз (тест)')


plt.legend()
plt.xlabel('Месяцы')
plt.ylabel('Количество пассажиров')
plt.title('Прогнозирование количества международных авиапассажиров')
plt.show()
