# подготовка данных
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import accuracy_score, classification_report


# get the wine dataset from sklearn and take a look at the description provided
wine = datasets.load_wine()
print(wine.DESCR)  # Вывод данных о составе базы

# обучение с учителем
# определяем переменные датасета
X = wine.data
y = wine.target

# Разделяем данные на обучающую и тестовую части
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучаем модель логистической регрессии
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Предсказываем результаты на тестовых данных
y_pred = model.predict(X_test)

# Оцениваем качество модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# обучение без учителя
# Обучаем модель K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Предсказываем кластеры
clusters = kmeans.predict(X)
print(X_test[:,0])
print(X_test[:,1])
print(clusters)
print(len(clusters))

# Визуализируем результаты кластеризации
print(len(X[:, 0]))
print(len(X[:, 1]))
print(len(clusters))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x')
plt.title('Clustering of Wine Dataset \n "x" - Test dates')
plt.xlabel('% of Alcohole')  # алкоголь, 1 столбец, потому х:0
plt.ylabel('Malic Acid')  # кислотность, 2 столбец, потому х:1
plt.show()
