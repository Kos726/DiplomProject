# подготовка данных
import numpy as np

from sklearn.model_selection import train_test_split
# from sklearn.datasets import fetch_openml
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report  # accuracy_score
from sklearn import datasets

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt

# get the wine dataset from sklearn and take a look at the description provided
wine = datasets.load_wine()
# print(wine.DESCR)  # Вывод данных о составе базы

# обучение с учителем
# определяем переменные датасета
X = wine.data
y = wine.target

# Разделяем данные на обучающую и тестовую части
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Тензоры полных данных
X_tensor = torch.from_numpy(X).float()
y_tensor = torch.from_numpy(y).long()
# Тензоры обучения
train_X_tensor = torch.from_numpy(X_train).float()
train_y_tensor = torch.from_numpy(y_train).long()
# Тензоры тестирования
test_X_tensor = torch.from_numpy(X_test).float()
test_y_tensor = torch.from_numpy(y_test).long()

# print(train_X_tensor.shape)
# print(train_y_tensor.shape)
assert (train_X_tensor.shape == X_train.shape)
assert (train_y_tensor.shape == y_train.shape)

# Массивы полных, тренировочных и тестовых данных
data_full = TensorDataset(X_tensor, y_tensor)
data_train = TensorDataset(train_X_tensor, train_y_tensor)
data_test = TensorDataset(test_X_tensor, test_y_tensor)

# Создаем загрузчик данных
train_minibatch = DataLoader(data_train, batch_size=65, shuffle=True)


# Определяем и создаем сеть
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.nodes = []
        # y = Ax + b s.t. 13 -> 96
        n0, n1 = 13, 96
        self.nodes.append(n0 * n1 + n1)
        self.fc1 = nn.Linear(n0, n1)  # слой 1
        # y = Ax + b s.t. 96 -> 120
        n1, n2 = n1, 120
        self.nodes.append(n1 * n2 + n2)
        self.fc2 = nn.Linear(n1, n2)  # слой 2
        # y = Ax + b s.t. 120 -> 120
        n2, n3 = n2, 120
        self.nodes.append(n1 * n3 + n3)
        self.fc3 = nn.Linear(n2, n3)  # слой 3
        # y = Ax + b s.t. 120 -> 90
        n3, n4 = n3, 90
        self.nodes.append(n1 * n4 + n4)
        self.fc4 = nn.Linear(n3, n4)  # слой 4
        # y = Ax + b s.t. 90 -> 76
        n4, n5 = n4, 76
        self.nodes.append(n1 * n5 + n5)
        self.fc5 = nn.Linear(n4, n5)  # слой 5
        # y = Ax + b s.t. 76 -> 45
        n5, n6 = n5, 45
        self.nodes.append(n1 * n6 + n6)
        self.fc6 = nn.Linear(n5, n6)  # слой 6
        # y = Ax + b s.t. 45 -> 22
        n6, n7 = n6, 22
        self.nodes.append(n1 * n7 + n7)
        self.fc7 = nn.Linear(n6, n7)  # слой 7
        # y = Ax + b s.t. 22 -> 10
        n7, n8 = n7, 10
        self.nodes.append(n1 * n8 + n8)
        self.fc8 = nn.Linear(n7, n8)  # слой 8
        # y = Ax + b s.t. 10 -> 3
        n8, n9 = n8, 3
        self.nodes.append(n1 * n9 + n9)
        self.fc9 = nn.Linear(n8, n9)  # слой 9

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x)
        return F.log_softmax(x, -1)

    def summary(self):
        for i in range(len(self.nodes)):
            print(f"| Shape: {i} | Nodes: {self.nodes[i]}")


# Подключаем сеть к модели
model = Net()
# Вывод узлов в сети
model.summary()

# Перекрестная энтропия
criterion = nn.CrossEntropyLoss()

# Стохастический градиентный спуск
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Цикл обучения
for epoch in range(1600):
    total_loss = 0
    for X_train, y_train in train_minibatch:
        # Построение графа
        X_train, y_train = Variable(X_train), Variable(y_train)
        # Оптимизация
        optimizer.zero_grad()
        # Расчет выходных данных
        output = model(X_train)
        # Вычисление ошибки
        loss = criterion(output, y_train)
        # Возврат ошибки
        loss.backward()
        # оптимизация
        optimizer.step()
        # Накопление ошибок
        total_loss += loss.item()

    # Отображение совокупных просчетов
    if (epoch + 1) % 20 == 0:
        print(epoch + 1, total_loss)


def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
    prediction = np.argmax(predictions[0].detach())
    # print("Input:", input)
    # print("Target:", target.numpy())
    # print("Prediction:", prediction.numpy())
    return prediction.numpy()


y_pred = []
for i in range(len(data_test)):
    input, target = data_test[i]
    p = predict_single(input, target, model).item()
    y_pred.append(p)

# Оцениваем качество модели
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Визуальная оценка результатов
# print(len(y_test))
# print(len(y_pred))
print("Классы тестовых данных:\n", y_test)
print("Классы предсказываемых данных:\n", np.array(y_pred))

# Предсказываем результаты на основных данных
# class_result = model_mlp.predict(X)

clusters = []
for i in range(len(data_full)):
    input, target = data_full[i]
    p = predict_single(input, target, model).item()
    clusters.append(p)

# print(len(np.array(train_y_tensor)))
# print(len(clusters))
print("Классы основных данных:\n", np.array(train_y_tensor))
print("Классы предсказываемых данных:\n", np.array(clusters))

# Визуализируем результаты кластеризации
# print(X[:, 0])
# print(X[:, 1])
# print(len(X[:, 0]))
# print(len(X[:, 1]))
# print(len(clusters))

plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x')
plt.title('Clustering of Wine Dataset \n "x" - Test dates')
plt.xlabel('% of Alcohole')  # алкоголь, 1 столбец, потому х:0
plt.ylabel('Malic Acid')  # кислотность, 2 столбец, потому х:1
plt.show()
