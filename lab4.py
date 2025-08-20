import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 1, 1]
], dtype=float)

y = np.array([[0], [1], [1], [0]], dtype=float)



def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_deriv(x):
    s = logistic(x)
    return s * (1 - s)



def squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def squared_error_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


def main():

    np.random.seed(42)
    input_dim = X.shape[1]
    output_dim = 1
    weights = np.random.randn(input_dim, output_dim)
    bias = np.zeros((1, output_dim))

    lr = 0.1
    epochs = 1000
    loss_history = []

    for ep in range(epochs):

        z = np.dot(X, weights) + bias
        activation = logistic(z)

        loss = squared_error(y, activation)
        loss_history.append(loss)


        d = squared_error_grad(y, activation)
        a = logistic_deriv(z)
        L = d *a

        grad_w = np.dot(X.T, L)
        grad_b = np.sum(L, axis=0, keepdims=True)


        weights -= lr * grad_w
        bias -= lr * grad_b

        if (ep + 1) % 100 == 0:
            print(f"Step {ep+1} → Current Error: {loss:.6f}")


    print("\nPredictions after training:")
    print(activation)

    plt.plot(loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.title("Loss Curve")
    plt.show()

if __name__=="__main__":
    main()