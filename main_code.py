import numpy as np
from tensorflow.keras.datasets import mnist

# -------------------- Reproducibility --------------------
np.random.seed(42)

# -------------------- Load Data --------------------
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 784).T / 255.0
X_test = X_test.reshape(X_test.shape[0], 784).T / 255.0

def one_hot(y, num_classes=10):
    onehot = np.zeros((num_classes, y.size))
    onehot[y, np.arange(y.size)] = 1
    return onehot

Y_train = one_hot(y_train)
Y_test = one_hot(y_test)

# -------------------- Activations --------------------
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(int)

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

# -------------------- Network Architecture --------------------
layer_sizes = [784, 64, 32, 10]

# -------------------- Parameter Initialization --------------------
def parameters(layer_sizes):
    param = {}
    for i in range(1, len(layer_sizes)):
        param[f"w{i}"] = (
            np.random.randn(layer_sizes[i], layer_sizes[i-1])
            * np.sqrt(2 / layer_sizes[i-1])
        )
        param[f"b{i}"] = np.zeros((layer_sizes[i], 1))
    return param

# -------------------- Forward Propagation --------------------
def forward_propagation(param, X):
    caches = {}
    A = X
    L = len(layer_sizes) - 1

    for i in range(1, L):
        Z = np.dot(param[f"w{i}"], A) + param[f"b{i}"]
        A = relu(Z)
        caches[f"Z{i}"] = Z
        caches[f"A{i}"] = A

    Z = np.dot(param[f"w{L}"], A) + param[f"b{L}"]
    A = softmax(Z)
    caches[f"Z{L}"] = Z
    caches[f"A{L}"] = A

    return A, caches

# -------------------- Backpropagation --------------------
def compute_gradients(param, caches, X, Y):
    grads = {}
    m = X.shape[1]
    L = len(layer_sizes) - 1

    A_final = caches[f"A{L}"]
    dZ = A_final - Y
    A_prev = caches[f"A{L-1}"]

    grads[f"dW{L}"] = np.dot(dZ, A_prev.T) / m
    grads[f"db{L}"] = np.sum(dZ, axis=1, keepdims=True) / m

    dA_prev = np.dot(param[f"w{L}"].T, dZ)

    for i in range(L-1, 0, -1):
        Z = caches[f"Z{i}"]
        A_prev = X if i == 1 else caches[f"A{i-1}"]

        dZ = dA_prev * relu_derivative(Z)
        grads[f"dW{i}"] = np.dot(dZ, A_prev.T) / m
        grads[f"db{i}"] = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA_prev = np.dot(param[f"w{i}"].T, dZ)

    return grads

# -------------------- Training (MINI-BATCH ADDED) --------------------
learning_rate = 0.01
num_epochs = 30        # reduced (mini-batch converges faster)
batch_size = 64
eps = 1e-9

param = parameters(layer_sizes)
m = X_train.shape[1]

for epoch in range(num_epochs):
    perm = np.random.permutation(m)
    X_shuffled = X_train[:, perm]
    Y_shuffled = Y_train[:, perm]

    for i in range(0, m, batch_size):
        X_batch = X_shuffled[:, i:i+batch_size]
        Y_batch = Y_shuffled[:, i:i+batch_size]

        A_final, caches = forward_propagation(param, X_batch)
        grads = compute_gradients(param, caches, X_batch, Y_batch)

        for l in range(1, len(layer_sizes)):
            param[f"w{l}"] -= learning_rate * grads[f"dW{l}"]
            param[f"b{l}"] -= learning_rate * grads[f"db{l}"]

    A_full, _ = forward_propagation(param, X_train)
    loss = -np.sum(Y_train * np.log(A_full + eps)) / m
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# -------------------- Prediction --------------------
def predict(param, X):
    A, _ = forward_propagation(param, X)
    return np.argmax(A, axis=0)

# -------------------- Accuracy --------------------
train_preds = predict(param, X_train)
train_acc = np.mean(train_preds == y_train)
print("Train Accuracy:", train_acc)

test_preds = predict(param, X_test)
test_acc = np.mean(test_preds == y_test)
print("MNIST Test Accuracy:", test_acc)
