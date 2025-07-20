# GEO-PROMO CLASSIFIER
# ------------------------------------------
# Prediksi apakah pelanggan berada dalam zona promosi restoran (radius 1 km)

import numpy as np
import matplotlib.pyplot as plt

# 1. Generate Data
np.random.seed(1)
m = 400  # jumlah titik
X = np.random.uniform(-1.5, 1.5, (2, m))  # 2 fitur: x dan y
X = X.T

# 2. Buat label y: 1 jika titik dalam lingkaran radius 1
Y = (X[:, 0]**2 + X[:, 1]**2 <= 1.0).astype(int)
Y = Y.reshape(-1, 1)

# 3. Visualisasi data
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=Y[:,0], cmap=plt.cm.Spectral)
plt.title("Data Pelanggan dalam/luar zona promo (lingkaran)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.show()

# 4. Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 5. Model Logistic Regression
def model(X, Y, lr=0.1, epochs=1000):
    m = X.shape[0]
    X_new = np.hstack([X, np.ones((m, 1))])  # tambah bias
    W = np.random.randn(3, 1) * 0.01

    losses = []
    for i in range(epochs):
        Z = np.dot(X_new, W)
        A = sigmoid(Z)
        cost = -np.mean(Y * np.log(A + 1e-8) + (1 - Y) * np.log(1 - A + 1e-8))
        dW = np.dot(X_new.T, (A - Y)) / m
        W -= lr * dW
        if i % 100 == 0:
            losses.append(cost)
            print(f"Epoch {i}: loss={cost:.4f}")

    return W

# 6. Train the model
W = model(X, Y, lr=0.5, epochs=1000)

# 7. Visualisasi decision boundary
def plot_decision_boundary(W, X):
    x_min, x_max = -1.5, 1.5
    y_min, y_max = -1.5, 1.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_new = np.hstack([grid, np.ones((grid.shape[0], 1))])
    probs = sigmoid(np.dot(grid_new, W)).reshape(xx.shape)

    plt.figure(figsize=(6,6))
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], cmap="RdYlBu", alpha=0.7)
    plt.scatter(X[:,0], X[:,1], c=Y[:,0], cmap=plt.cm.Spectral)
    plt.title("Decision Boundary GeoPromo Classifier")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.show()

plot_decision_boundary(W, X)
