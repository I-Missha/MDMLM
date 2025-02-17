import numpy as np
import matplotlib.pyplot as plt


# Функция для генерации выборки
def generate_sample(f, N, epsilon_0, error_type):
    x = np.random.uniform(-1, 1, N)  # Генерация аргументов x из равномерного распределения

    # Генерация ошибки измерения
    if error_type == 'uniform':
        epsilon = np.random.uniform(-epsilon_0, epsilon_0, N)
    elif error_type == 'normal':
        epsilon = np.random.normal(0, epsilon_0 / 2, N)  # Стандартное отклонение = ε0/2 для контроля диапазона

    y = f(x) + epsilon  # Добавление ошибки к значениям функции
    return x, y


# Первая функция: f(x) = ax^3 + bx^2 + cx + d, коэффициенты случайные из [-3, 3]
a, b, c, d = np.random.uniform(-3, 3, 4)
f1 = lambda x: a * x ** 3 + b * x ** 2 + c * x + d

# Вторая функция: f(x) = x sin(2πx)
f2 = lambda x: x * np.sin(2 * np.pi * x)

# Параметры генерации
N = 100  # Количество точек
epsilon_0_values = [0.1, 0.5]  # Разные значения ε0
error_types = ['uniform', 'normal']

# Визуализация
fig, axes = plt.subplots(len(epsilon_0_values), len(error_types), figsize=(10, 8))

for i, epsilon_0 in enumerate(epsilon_0_values):
    for j, error_type in enumerate(error_types):
        x1, y1 = generate_sample(f1, N, epsilon_0, error_type)
        x2, y2 = generate_sample(f2, N, epsilon_0, error_type)

        ax = axes[i, j]
        ax.scatter(x1, y1, color='blue', alpha=0.6, label='f1 samples')
        ax.scatter(x2, y2, color='red', alpha=0.6, label='f2 samples')

        # Отображение самих функций
        x_vals = np.linspace(-1, 1, 200)
        ax.plot(x_vals, f1(x_vals), color='blue', label='f1(x)')
        ax.plot(x_vals, f2(x_vals), color='red', label='f2(x)')

        ax.set_title(f"ε0={epsilon_0}, Error={error_type}")
        ax.legend()
        ax.grid(True)

plt.tight_layout()
plt.show()