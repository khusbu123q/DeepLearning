import numpy as np

def main():
    data = np.load('1000G_reqnorm_float64.npy')

    num_genes, num_samples = data.shape
    print(f"Number of genes (rows): {num_genes}")
    print(f"Number of samples (columns): {num_samples}")

    data_means = data.mean(axis=1, keepdims=True)
    data_stds = data.std(axis=1, keepdims=True) + 1e-3
    data = (data - data_means) / data_stds

    num_lm = 943  # Change this if needed
    X = data[:num_lm, :].T   # (samples × genes_lm)
    Y = data[num_lm:, :].T   # (samples × genes_other)

    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]

    train_end = int(0.7 * num_samples)
    val_end = int(0.9 * num_samples)

    X_train, Y_train = X[:train_end], Y[:train_end]
    X_val, Y_val = X[train_end:val_end], Y[train_end:val_end]
    X_test, Y_test = X[val_end:], Y[val_end:]

    np.save('1000G_X_train.npy', X_train)
    np.save('1000G_Y_train.npy', Y_train)

    np.save('1000G_X_val.npy', X_val)
    np.save('1000G_Y_val.npy', Y_val)

    np.save('1000G_X_test.npy', X_test)
    np.save('1000G_Y_test.npy', Y_test)

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

if __name__ == '__main__':
    main()
