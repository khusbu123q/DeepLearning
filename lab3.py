import numpy as np

def relu_function(x):
    return np.maximum(0, x)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def softmax_function(z):
    exp_vals = np.exp(z - np.max(z))
    return exp_vals / np.sum(exp_vals)


def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_grad(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)

def forward_pass(x, weights, biases):
    activate = [x]
    zfun = []

    a = x
    for layer in range(len(weights)):
        z = np.dot(weights[layer], a) + biases[layer]
        zfun.append(z)

        if layer < len(weights) - 1:
            a = relu_function(z)
        else:
            choice = input("Choose final activation (R=ReLU, S=Softmax): ")
            if choice.upper() == "S":
                a = softmax_function(z)
            elif choice.upper() == "R":
                a = relu_function(z)
            else:
                print("Invalid choice → using ReLU")
                a = relu_function(z)

        activate.append(a)

    return a, activate, zfun


def backward_pass(y_true, weights, biases, activate, zfun):

    grad_w = [None] * len(weights)
    grad_b = [None] * len(biases)

    delta = mse_grad(y_true, activate[-1])

    for layer in reversed(range(len(weights))):
        print(f"\nBackprop at Layer {layer+1}")


        local_grad = relu_derivative(zfun[layer])


        delta = delta * local_grad


        grad_w[layer] = np.outer(delta, activate[layer])
        grad_b[layer] = delta


        delta = np.dot(weights[layer].T, delta)

    return grad_w, grad_b


def main():

    n_inputs = int(input("Enter number of inputs: "))
    x = np.array([float(input(f"Input {i+1}: ")) for i in range(n_inputs)])


    n_layers = int(input("Enter number of layers: "))
    layer_sizes = [n_inputs] + [int(input(f"Neurons in layer {i+1}: ")) for i in range(n_layers)]


    weights = [np.random.randn(layer_sizes[i+1], layer_sizes[i]) for i in range(n_layers)]
    biases = [np.random.randn(layer_sizes[i+1]) for i in range(n_layers)]


    output, activs, zs = forward_pass(x, weights, biases)
    print("Final Network Output:", output)


    y_true = np.array([float(input(f"Expected output {i+1}: ")) for i in range(len(output))])


    grad_w, grad_b = backward_pass(y_true, weights, biases, activs, zs)


    print("\nGradients (Weights):")
    for g in grad_w:
        print(g)

    print("\nGradients (Biases):")
    for g in grad_b:
        print(g)


if __name__ == "__main__":
    main()
