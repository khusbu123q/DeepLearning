import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    sig_z = 1 / (1 + np.exp(-z))
    return np.array(sig_z), np.array(sig_z * (1 - sig_z)),np.mean(sig_z)


def tanh(z):
    tanh_z = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    return np.array(tanh_z), np.array(1 - tanh_z ** 2),np.mean(tanh_z)


def relu(z):
    ReLU = (z + np.abs(z)) / 2
    der_ReLU = 0.5 + 0.5 * np.sign(z)
    return np.array(ReLU), np.array(der_ReLU),np.mean(ReLU)


def leaky_relu(z):
    leaky_ReLU = np.maximum(0.01 * z, z)
    der_leaky_ReLU = np.where(z > 0, 1, 0.01)
    return np.array(leaky_ReLU), np.array(der_leaky_ReLU),np.mean(leaky_ReLU)


def softmax(z):
    denominator = sum(np.exp(val) for val in z)
    numerator = np.exp(z)
    softmax = numerator / denominator
    jacobian = []
    for i in range(len(z)):
        temp = []
        for j in range(len(z)):
            if i == j:
                temp.append(softmax[i] * (1 - softmax[i]))
            else:
                temp.append(-softmax[i] * softmax[j])
        jacobian.append(temp)
    return np.array(softmax), np.array(jacobian),np.mean(softmax)

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

    sft, derv_sft, mean_sft = softmax(z)
    print("Mean of softmax: ",mean_sft)
    plt.figure(figsize=(8, 6))
    plt.plot(z, sft, label="Softmax function")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()
    print("Softmax output:", sft)
    print("Softmax Jacobian:")
    print(derv_sft)

if __name__ == "__main__":
    main()