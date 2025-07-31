import numpy as np
import random

def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    s = np.sum(np.exp(z))
    return [float(np.exp(i) / s) for i in z]

def forward_pass(x, w, b):
    a = x
    for j in range(len(w)):
        z = np.dot(w[j], a) + b[j]
        print(f"\nLayer {j + 1}")
        print("Weights:\n", w[j])
        print("Biases:\n", b[j])
        print("Z (Weighted sum):\n", z)

        # For hidden layers use ReLU
        if j < len(w) - 1:
            a = ReLU(z)
        else:
            while True:
                print("Choose output activation:")
                print("1. Type R - for ReLU")
                print("2. Type S - for Softmax")
                prompt = input("Enter R/S: ").strip().upper()
                if prompt == "S":
                    a = softmax(z)
                    break
                elif prompt == "R":
                    a = ReLU(z)
                    break
                else:
                    print("Please enter a valid option (R or S).")
    return a

def layers(num_layers, input_size):
    neu = []
    for k in range(num_layers):
        n = int(input(f"Enter the number of neurons in layer {k+1}: "))
        neu.append(n)
    neu.insert(0, input_size)
    return neu

def weights(n, num_layers):
    w = []
    b = []
    choice = input("\nDo you want to manually enter weights and biases? (y/n): ").strip().lower()
    for i in range(num_layers):
        if choice == "y":
            print(f"\nEnter weights for Layer {i+1} ({n[i+1]}x{n[i]}):")
            weight = []
            for j in range(n[i+1]):
                row = []
                for k in range(n[i]):
                    val = float(input(f"Weight[{j}][{k}]: "))
                    row.append(val)
                weight.append(row)
            weight = np.array(weight)

            print(f"Enter biases for Layer {i+1} (size {n[i+1]}):")
            bias = [float(input(f"Bias[{j}]: ")) for j in range(n[i+1])]
            bias = np.array(bias)
        else:
            # Random weights and biases
            weight = np.random.uniform(0.001, 1, (n[i+1], n[i]))
            bias = np.random.uniform(0.001, 1, n[i+1])
        w.append(weight)
        b.append(bias)
    return w, b

def main():
    q = int(input("Enter the number of inputs: "))
    x = np.array([float(input(f"Input {i+1}: ")) for i in range(q)])
    l = int(input("Enter the number of layers: "))
    n = layers(l, len(x))
    w, b = weights(n, l)
    output = forward_pass(x, w, b)
    print("\nNetwork Output:", output)

if __name__ == "__main__":
    main()
