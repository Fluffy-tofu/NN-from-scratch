"""
Very simple feedforward NN

Input-Layer:
    5-Dimensional Input layer

Hidden-Layer:
    three neurons

Output-Layer:
    1-Dimensional Output layer

Initializing of values:
    W uses the Xavier Initialization to create an initial value for the weights
    W ∼ N(0, 2/(n_in + n_out))
    W is represented as a (5, 3) matrix
    n_in is the number of input features and
    n_out is the number of neurons in the layer
    N is the normal distribution function


Forward Propagation:
    Z = W.T * X + b

Activation Function:
    Sigmoid: A = 1/(1+e^-Z)
    ReLU A = max(0, 7)

Loss Function:
    L = 1/2 * (A - Y)^2

Backpropagation:
    Derivatives:
        dL/dA = (A - Y)
        dA/dZ = A(1 - A)
        dZ/dW = X
        dZ/db = 1

    Chain rule:
        dL/dW = dL/dA * dA/dZ * dZ/dW = (A − Y) * A(1 − A) * X
        dL/db = dL/dA * dA/dZ * dZ/db = (A − Y) * A(1 − A)

        In code:
            dL_dW = np.dot(X, (A − Y) * A(1 − A).T)
            -> n.dot gets used due to differences in the shapes between X and the rest
            dL_db = (A − Y) * A(1 − A)

    Gradient Descent:
        W = W - n * dL/dW
        b = b - n * dL/db
        where n ist the learning rate
"""


import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def feedforward_nn(epochs, x, y):
    n_in = 5
    n_out = 3
    w = np.random.normal(0, np.sqrt(2 / (n_in + n_out)), size=(n_in, n_out))
    b = np.zeros((n_out, 1))
    learning_rate = 0.005
    for epoch in range(epochs):
        z = np.dot(w.T, x) + b
        a = sigmoid(z)

        delta = (a - y) * a * (1 - a)
        dl_dw = np.dot(x, delta.T)
        dl_db = delta

        w = w - learning_rate * dl_dw
        b = b - learning_rate * dl_db

        if epoch % 100 == 0:
            w_formatted = [f'{weight:.3f}' for weight in w.flatten()]

            b_formatted = [f'{bias:.3f}' for bias in b.flatten()]

            loss = ((a - y) ** 2).mean()
            loss_formatted = f'{loss:.3f}'

            print(f"Epoch {epoch}: w = {w_formatted}, b = {b_formatted}, loss = {loss_formatted}")

    return w, b, loss


def test_nn():
    x = np.random.rand(5,1)
    y = np.array([0.5])

    epochs = 1000

    w, b, loss = feedforward_nn(epochs, x, y)

    print(f"Final weights: {w.flatten()}")
    print(f"Final bias: {b}")
    print(f"Final loss: {loss}")

test_nn()
