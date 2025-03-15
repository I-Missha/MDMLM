import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import time

# генериуем двумерное нормальное распределение с шумом -> делаем сдвиг
def generate_xor(n_samples=100, noise=0.1):
    X = np.random.randn(n_samples, 2) * noise
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if i % 4 == 0:
            X[i] += [0, 0]
        elif i % 4 == 1:
            X[i] += [0, 1]
            y[i] = 1
        elif i % 4 == 2:
            X[i] += [1, 0]
            y[i] = 1
        else:
            X[i] += [1, 1]
    return X, y

# как обсуждалось в конце семинара, генерируем выборку при помощи параметризации
def generate_spirals(n_samples=100, noise=0.1):
    n = n_samples // 2
    theta = np.linspace(0, 2 * np.pi, n)  # генерируем точки от 0 до 2pi
    r = np.linspace(0.5, 2, n)  # pflftv hflbec Начинаем с 0.5 чтобы избежать пересечения в центре
    # Первая спираль
    X0 = np.column_stack([r * np.sin(theta), r * np.cos(theta)]) + np.random.normal(0, noise, (n, 2))
    # Вторая спираль со смещением
    X1 = np.column_stack([r * np.sin(theta + np.pi), r * np.cos(theta + np.pi)]) + np.random.normal(0, noise, (n, 2))
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y


datasets = {
    'moons': make_moons(n_samples=100, noise=0.2, random_state=0),
    'xor': generate_xor(),
    'spirals': generate_spirals(),
    'gaussian': make_blobs(n_samples=100, centers=2, random_state=0)
}

# ступенчатый персептрон
class ThresholdPerceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size)
        self.bias = 0.0

    # 1 если скалярное произведение + сдвиг > 0, иначе 0
    def predict(self, X):
        return np.array([1 if (np.dot(x, self.weights) + self.bias) >= 0 else 0 for x in X])
    # о
    def train(self, X, y, lr=0.1, epochs=100):
        for _ in range(epochs):
            converged = True
            for xi, yi in zip(X, y):
                y_pred = 1 if (np.dot(xi, self.weights) + self.bias) >= 0 else 0
                error = yi - y_pred
                if error != 0:
                    self.weights += lr * error * xi
                    self.bias += lr * error
                    converged = False
            if converged:
                break


class SigmoidPerceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return (self.sigmoid(z) >= 0.5).astype(int)

    def train(self, X, Y, lr=0.01, epochs=1000):
        m = len(Y)
        X = np.array(X)
        y = np.array(Y)
        for _ in range(epochs):
            # calc prediction
            z = np.dot(X, self.weights) + self.bias
            a = self.sigmoid(z)
            dz = a - y # в силу того, что персептрон однослойный формула ошибки получается такой
            dw = (1 / m) * np.dot(X.T, dz) # считаем сразу градиент для всей выборки см формулу из отчета
            db = (1 / m) * np.sum(dz)
            self.weights -= lr * dw # сдвигаем веса
            self.bias -= lr * db


def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5 # выбираем границы для отображения
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(title)
    plt.show()


for name, (X, Y) in datasets.items():
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


    tp = ThresholdPerceptron(input_size=2)
    start = time.time()
    tp.train(X_train, y_train, epochs=10000)
    tp_time = time.time() - start
    y_pred_tp = tp.predict(X_test)
    cm_tp = confusion_matrix(y_test, y_pred_tp)


    sp = SigmoidPerceptron(input_size=2)
    start = time.time()
    sp.train(X_train, y_train, epochs=10000)
    sp_time = time.time() - start
    y_pred_sp = sp.predict(X_test)
    cm_sp = confusion_matrix(y_test, y_pred_sp)

    print(f"\nDataset: {name}")
    print("Threshold Perceptron:")
    print(f"Confusion Matrix:\n{cm_tp}")
    print(f"Training Time: {tp_time:.4f}s")

    print("\nSigmoid Perceptron:")
    print(f"Confusion Matrix:\n{cm_sp}")
    print(f"Training Time: {sp_time:.4f}s")


    plot_decision_boundary(tp, X, Y, f'Threshold Perceptron ({name})')
    plot_decision_boundary(sp, X, Y, f'Sigmoid Perceptron ({name})')