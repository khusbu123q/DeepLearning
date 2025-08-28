import numpy as np

def quantile_normalize(data):
    sorted_idx = np.argsort(data, axis=0)
    sorted_data = np.sort(data, axis=0)
    mean_values = np.mean(sorted_data, axis=1)
    normalized_data = np.zeros_like(data)
    for col in range(data.shape[1]):
        normalized_data[sorted_idx[:, col], col] = mean_values
    return normalized_data

# Load
data = np.load("1000G_float64.npy")

# Landmark genes (first 943 rows)
landmark = data[:943, :]
landmark_norm = quantile_normalize(landmark)

# Target genes (remaining rows)
target = data[943:, :]
target_norm = quantile_normalize(target)

# Combine back
data_reqnorm = np.vstack((landmark_norm, target_norm))

print(f"Final normalized matrix shape: {data_reqnorm.shape}")

# Save
np.save("1000G_reqnorm_float64.npy", data_reqnorm)

def main():
    # Load data
    data = np.load('1000G_reqnorm_float64.npy')

    # Print shape info
    num_genes, num_samples = data.shape
    print(f"Number of genes (rows): {num_genes}")
    print(f"Number of samples (columns): {num_samples}")

    # Normalize each row (gene)
    data_means = data.mean(axis=1, keepdims=True)
    data_stds = data.std(axis=1, keepdims=True) + 1e-3
    data = (data - data_means) / data_stds

    # Number of landmark genes
    num_lm = 943  # Change this if needed
    X = data[:num_lm, :].T
    Y = data[num_lm:, :].T

    # Save full X and Y
    np.save('1000G_X_float64.npy', X)
    np.save('1000G_Y_float64.npy', Y)

    # Save Y in two parts
    mid = Y.shape[1] // 2
    np.save(f'1000G_Y_0-{mid}_float64.npy', Y[:, :mid])
    np.save(f'1000G_Y_{mid}-{Y.shape[1]}_float64.npy', Y[:, mid:])

if __name__ == '__main__':
    main()


