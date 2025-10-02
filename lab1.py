import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    sig = 1 / (1 + np.exp(-z))
    return np.array(sig), np.array(sig * (1 - sig)),np.mean(sig)


def tanh(z):
    tanh = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return np.array(tanh), np.array(1 - tanh ** 2),np.mean(tanh)


def relu(z):
    Re = (z + np.abs(z)) / 2
    der = 0.5 + 0.5 * np.sign(z)
    return np.array(Re), np.array(der),np.mean(Re)


def leaky_relu(z):
    leaky= np.maximum(0.01 * z, z)
    der_leaky= np.where(z > 0, 1, 0.01)
    return np.array(leaky), np.array(der_leaky),np.mean(leaky)


def softmax(z):
    denominator = sum(np.exp(val) for val in z)
    numerator = np.exp(z)
    sx = numerator / denominator
    jacobian = []
    for i in range(len(z)):
        temp = []
        for j in range(len(z)):
            if i == j:
                temp.append(sx[i] * (1 - sx[i]))
            else:
                temp.append(-sx[i] * sx[j])
        jacobian.append(temp)
    return np.array(sx), np.array(jacobian),np.mean(sx)

def plots(z, func_name, func):
    y, dy, mean = func(z)
    print(f'Mean of the {func_name} output: {mean}')
    plt.figure(figsize=(8, 6))
    plt.plot(z, y, label=f"{func_name} function")
    plt.plot(z, dy, label=f"Derivative of {func_name}")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

def main():
    z = np.linspace(-10, 10, 100)

    plots(z, "Sigmoid", sigmoid)
    plots(z, "Tanh", tanh)
    plots(z, "ReLU", relu)
    plots(z, "Leaky ReLU", leaky_relu)

    s, derv_s, mean_s = softmax(z)
    print("Mean of softmax: ",mean_s)
    plt.figure(figsize=(8, 6))
    plt.plot(z, s, label="Softmax function")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
    print("Softmax output:", s)
    print("Softmax Jacobian:")
    print(derv_s)

if __name__ == "__main__":
    main()
