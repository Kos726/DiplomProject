# подготовка данных
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn import datasets
import matplotlib.pyplot as plt

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# для корректной работы кода на персональном комьютере
# необходимо включение параметра "os.environ", так что
# его включение происходит до подключения "tensorflow.keras"
# поэтому вынужденное нарушение PEP8
from tensorflow.keras import utils, models
from tensorflow.keras.layers import Input, Dense, Dropout


# get the wine dataset from sklearn and take a look at the description provided
wine = datasets.load_wine()
print(wine.DESCR)  # Вывод данных о составе базы

# обучение с учителем
# определяем переменные датасета
X = wine.data
y = wine.target

# Разделяем данные на обучающую и тестовую части
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Создание и обучение модели k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred_knn = knn.predict(X_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f'Accuracy of k-NN: {accuracy_knn:.4f}')

# Предобработка меток для MLP
y_train_mlp = utils.to_categorical(y_train, 15)
y_test_mlp = utils.to_categorical(y_test, 15)

# Поля для визуальной сверки данных
"""
print(len(y_train_mlp))
print(len(y_test_mlp))
print(y_train_mlp)
print(y_test_mlp)
"""

# Создание модели MLP
type_model = 3  # 1, 2, 3, 4, 5

params_ = X_train.shape[1]

if type_model == 5:
    model_mlp = models.Sequential([
        Input(shape=(params_, )),
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
        Dense(5, activation='linear'),
        Dense(15, activation='softmax'),
    ])
    batch_size_ = 1  # 1
    validation_split_ = 0.3  # 0.3 - В процесссе настройки модели пришлось отказаться от параметра
    epochs_ = 110

elif type_model == 2:
    model_mlp = models.Sequential([
        Input(shape=(params_, )),
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
        Dense(5, activation='relu'),
        Dense(15, activation='softmax'),  # softmax  sigmoid
    ])
    batch_size_ = None  # 3
    validation_split_ = 0.25  # 0.25 В процесссе настройки модели пришлось отказаться от параметра
    epochs_ = 110

elif type_model == 3:
    model_mlp = models.Sequential([
        Input(shape=(params_, )),
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
        Dense(units=22, activation='linear'),
        Dense(15, activation='softmax'),  # softmax  sigmoid
    ])
    batch_size_ = 1  # 3
    validation_split_ = 0.2  # 0.05, 0.25 В процесссе настройки модели пришлось отказаться от параметра
    epochs_ = 110  # 85

elif type_model == 4:
    model_mlp = models.Sequential([
        Input(shape=(params_, )),
        Dense(units=28, activation='relu'),  # 17, 58
        Dense(units=38, activation='relu'),  # 32, 58, 108
        Dense(units=52, activation='relu'),  # 35, 58, 102
        Dense(units=45, activation='relu'),  # 42, 58, 125
        Dropout(0.2),
        Dense(units=45, activation='relu'),  # 40, 91, 125
        Dense(units=40, activation='relu'),  # 42, 60, 125
        Dropout(0.2),
        Dense(units=42, activation='relu'),  # 40, 89, 125
        Dense(units=40, activation='relu'),  # 33, 48, 100
        Dense(units=22, activation='relu'),  # 22, 42, 52
        Dense(15, activation='softmax'),  # softmax  sigmoid
    ])
    batch_size_ = 1  # 1
    validation_split_ = 0.2  # 0.25 В процесссе настройки модели пришлось отказаться от параметра
    epochs_ = 110

elif type_model == 5:
    model_mlp = models.Sequential([
        Input(shape=(params_, )),
        Dense(10, activation='relu'),
        Dense(78, activation='relu'),
        Dense(15, activation='softmax'),   # softmax  sigmoid
    ])
    batch_size_ = 1  # 5
    validation_split_ = 0.25  # 0.25 В процесссе настройки модели пришлось отказаться от параметра
    epochs_ = 90

model_mlp.summary()

# Компиляция и обучение модели MLP
model_mlp.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cl = model_mlp.fit(X_train, y_train_mlp,
                   batch_size=batch_size_,
                   validation_data=(X_test, y_test_mlp),  # validation_split=validation_split_,
                   epochs=epochs_)

# Оценка модели на тестовой выборке
loss_mlp, accuracy_mlp = model_mlp.evaluate(X_test, y_test_mlp)
print(f'Accuracy of MLP: {accuracy_mlp:.4f}')

# Предсказываем результаты на тестовых данных
predict = model_mlp.predict(X_test)
y_pred = []
for i in range(len(predict)):
    y_pred.append(np.argmax(predict[i]))

# Оцениваем качество модели
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Визуальная оценка результатов
# print(predict.round(decimals=1))
# print(len(y_test))
# print(len(y_pred))
print("Классы тестовых данных:", y_test)
print("Классы предсказываемых данных:", np.array(y_pred))

fig, ax = plt.subplots(figsize=(8, 3))

plt.plot(cl.history['accuracy'], label='accuracy')
plt.plot(cl.history['val_accuracy'], label='val_accuracy', linestyle='--')
plt.plot(cl.history['loss'], label='loss')
plt.plot(cl.history['val_loss'], label='val_loss', linestyle='--')
plt.xlim(10, 120)
plt.ylim(0, 2)
plt.legend()
plt.show()

# Предсказываем результаты на основных данных
class_result = model_mlp.predict(X)

clusters = []
for i in range(len(class_result)):
    t_ = np.argmax(class_result[i]).round(decimals=1)
    t = np.int32(t_).item()
    clusters.append(t)

# print(clusters)

# Визуализируем результаты кластеризации
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x')
plt.title('Clustering of Wine Dataset \n "x" - Test dates')
plt.xlabel('% of Alcohole')  # алкоголь, 1 столбец, потому х:0
plt.ylabel('Malic Acid')  # кислотность, 2 столбец, потому х:1
plt.show()
