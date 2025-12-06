import numpy as np

class Layer:
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def feedforward(self, x):
        self.x = x
        self.z = x @ self.W + self.b
        self.a = self.sigmoid(self.z)
        return self.a

    def backward(self, grad_output, lr):
        # gradient wrt z
        dz = grad_output * self.sigmoid_deriv(self.z)

        # gradients wrt parameters
        dW = self.x.T @ dz
        db = dz

        # gradient wrt input
        dx = dz @ self.W.T

        # update parameters
        self.W -= lr * dW
        self.b -= lr * db

        return dx


class OurNeuralNetwork:
    def __init__(self):
        self.h = Layer(2, 2)   # hidden layer: 2 inputs → 2 neurons
        self.o = Layer(2, 1)   # output layer: 2 inputs → 1 neuron

    def feedforward(self, x):
        x = x.reshape(1, 2)
        out_h = self.h.feedforward(x)
        out_o = self.o.feedforward(out_h)
        return out_o[0, 0]

    def train(self, X, y, lr=0.1, epochs=5000):
        for epoch in range(epochs):
            for xi, yi in zip(X, y):

                xi = xi.reshape(1, 2)

                # forward pass
                h_out = self.h.feedforward(xi)
                y_pred = self.o.feedforward(h_out)

                # loss gradient wrt y_pred
                grad_y_pred = -2 * (yi - y_pred)
                grad_y_pred = np.array([[grad_y_pred]])

                # backprop
                grad_h = self.o.backward(grad_y_pred, lr)
                self.h.backward(grad_h, lr)

            # print loss occasionally
            if epoch % 500 == 0:
                preds = np.array([self.feedforward(xi) for xi in X])
                mse = np.mean((y - preds)**2)
                print(f"Epoch {epoch}   Loss = {mse:.4f}")


# ---------------------------
# TRAIN ON XOR
# ---------------------------

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 1, 1, 0])

nn = OurNeuralNetwork()
nn.train(X, y, lr=0.5, epochs=5000)

print("\nFinal Predictions:")
for xi in X:
    print(xi, " → ", nn.feedforward(xi))
