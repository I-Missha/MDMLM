import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Зафиксируем генератор случайных чисел для воспроизводимости (необязательно)
np.random.seed(42)


# 1. Генерация данных
def true_function(x):
    return x * np.sin(2 * np.pi * x)


N = 30  # Размер обучающей выборки
N_test = 100  # Размер тестовой выборки

# Генерируем точки из равномерного распределения на [-1, 1]
X_train = np.random.uniform(-1, 1, N)
X_test = np.random.uniform(-1, 1, N_test)

# Задаём амплитуду шума
epsilon_0 = 0.2
# Ошибка - равномерно распределённая на отрезке [-epsilon_0, epsilon_0]
noise_train = np.random.uniform(-epsilon_0, epsilon_0, N)
noise_test = np.random.uniform(-epsilon_0, epsilon_0, N_test)

# Формируем целевые значения
y_train = true_function(X_train) + noise_train
y_test = true_function(X_test) + noise_test

# Преобразуем X в двумерный массив для sklearn
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# 2. Обучение полиномиальных моделей с разными степенями
degrees = [1, 3, 9]

models = {}
train_errors = {}
test_errors = {}

for deg in degrees:
    # Pipeline: сначала генерируем полиномиальные признаки, затем линейная регрессия
    model = make_pipeline(PolynomialFeatures(deg), LinearRegression())
    model.fit(X_train, y_train)

    # Предсказания на обучающей и тестовой выборках
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Считаем MSE
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    models[deg] = model
    train_errors[deg] = mse_train
    test_errors[deg] = mse_test

# 3. Визуализация результатов
# Сгенерируем сетку для гладкого отрисовывания «истинной» функции и предсказаний
X_plot = np.linspace(-1, 1, 200).reshape(-1, 1)
y_true = true_function(X_plot.ravel())

plt.figure(figsize=(12, 8))

# Отрисуем исходные точки
plt.scatter(X_train, y_train, color='blue', alpha=0.6, label='Обучающая выборка')
plt.scatter(X_test, y_test, color='green', alpha=0.6, label='Тестовая выборка')

# График истинной функции (без шума)
plt.plot(X_plot, y_true, 'r--', label='Истинная функция f(x)')

# Отрисуем аппроксимации полиномиальных моделей
colors = ['magenta', 'orange', 'black']  # для наглядности
for deg, color in zip(degrees, colors):
    y_plot = models[deg].predict(X_plot)
    plt.plot(X_plot, y_plot, color=color,
             label=f'Полином степени {deg} (Train MSE={train_errors[deg]:.3f}, Test MSE={test_errors[deg]:.3f})')

plt.title("Полиномиальная регрессия: недообучение, переобучение и хороший баланс")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()
