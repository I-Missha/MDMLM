import numpy as np
import matplotlib.pyplot as plt

def generate_sample(f, N, epsilon_0, error_type):
    x = np.random.uniform(-1, 1, N)

    # Генерация ошибки измерения
    if error_type == 'uniform':
        epsilon = np.random.uniform(-epsilon_0, epsilon_0, N)
    elif error_type == 'normal':
        epsilon = np.random.normal(0, epsilon_0 / 2, N)

    y = f(x) + epsilon  # Добавление ошибки к значениям функции
    return x, y


a, b, c, d = np.random.uniform(-3, 3, 4)
func_a = lambda x: a * x ** 3 + b * x ** 2 + c * x + d

func_b = lambda x: x * np.sin(2 * np.pi * x)

# Параметры генерации
N = 100  # Количество точек
epsilon_0_values = [0.1, 0.5]  # Разные значения ε0
error_types = ['uniform', 'normal']

# Визуализация
fig, axes = plt.subplots(len(epsilon_0_values), len(error_types), figsize=(10, 8))

for i, epsilon_0 in enumerate(epsilon_0_values):
    for j, error_type in enumerate(error_types):
        x_a, y_a = generate_sample(func_a, N, epsilon_0, error_type)
        x_b, y_b = generate_sample(func_b, N, epsilon_0, error_type)

        ax = axes[i, j]
        ax.scatter(x_a, y_a, color='blue', alpha=0.6, label='func_a samples')
        ax.scatter(x_b, y_b, color='red', alpha=0.6, label='func_b samples')

        # Отображение самих функций
        x_vals = np.linspace(-1, 1, 200)
        ax.plot(x_vals, func_a(x_vals), color='blue', label='func_a(x)')
        ax.plot(x_vals, func_b(x_vals), color='red', label='func_b(x)')

        ax.set_title(f"ε0={epsilon_0}, Error={error_type}")
        ax.legend()
        ax.grid(True)

plt.tight_layout()
plt.show()