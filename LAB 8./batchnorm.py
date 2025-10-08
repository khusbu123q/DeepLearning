import numpy as np

class BatchNorm:
    def __init__(self, num_features, eps=1e-5, momentum=0.9, lr=0.01):
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.eps = eps
        self.lr = lr
        self.momentum = momentum
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def forward(self, x, training=True):
        self.x = x
        if training:
            self.batch_mean = np.mean(x, axis=0, keepdims=True)
            self.batch_var = np.var(x, axis=0, keepdims=True)
            self.x_norm = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
            out = self.gamma * self.x_norm + self.beta

            # update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma * x_norm + self.beta
        return out

    def backward(self, grad_op):
        N = self.x.shape[0]

        # Gradients wrt gamma and beta
        d_gamma = np.sum(grad_op * self.x_norm, axis=0, keepdims=True)
        d_beta = np.sum(grad_op, axis=0, keepdims=True)

        # Update params
        self.gamma -= self.lr * d_gamma
        self.beta -= self.lr * d_beta

        # Grad wrt input
        dx_norm = grad_op * self.gamma
        d_var = np.sum(dx_norm * (self.x - self.batch_mean) * -0.5 * (self.batch_var + self.eps) ** (-1.5),
                       axis=0, keepdims=True)
        d_mean = np.sum(dx_norm * -1 / np.sqrt(self.batch_var + self.eps), axis=0, keepdims=True) + \
                 d_var * np.mean(-2 * (self.x - self.batch_mean), axis=0, keepdims=True)
        dx = dx_norm / np.sqrt(self.batch_var + self.eps) + \
             d_var * 2 * (self.x - self.batch_mean) / N + \
             d_mean / N

        return dx

class LayerNorm:
    def __init__(self, num_features, eps=1e-5, lr=0.01):
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        self.eps = eps
        self.lr = lr

    def forward(self, x):
        self.x = x
        # mean and var across features (axis=1)
        self.mean = np.mean(x, axis=1, keepdims=True)
        self.var = np.var(x, axis=1, keepdims=True)
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        out = self.gamma * self.x_norm + self.beta
        return out

    def backward(self, grad_op):
        N, D = self.x.shape

        # Gradients wrt gamma and beta
        d_gamma = np.sum(grad_op * self.x_norm, axis=0, keepdims=True)
        d_beta = np.sum(grad_op, axis=0, keepdims=True)

        # Update params
        self.gamma -= self.lr * d_gamma
        self.beta -= self.lr * d_beta

        # Grad wrt input
        dx_norm = grad_op * self.gamma
        d_var = np.sum(dx_norm * (self.x - self.mean) * -0.5 * (self.var + self.eps) ** (-1.5), axis=1, keepdims=True)
        d_mean = np.sum(dx_norm * -1 / np.sqrt(self.var + self.eps), axis=1, keepdims=True) + \
                 d_var * np.mean(-2 * (self.x - self.mean), axis=1, keepdims=True)
        dx = dx_norm / np.sqrt(self.var + self.eps) + d_var * 2 * (self.x - self.mean) / D + d_mean / D

        return dx

def run_example():
    # New toy dataset: 5 samples, 4 features (different from your original)
    X = np.array([
        [0.5, 1.2, -0.3, 2.0],
        [1.0, 0.8,  0.0, 1.5],
        [0.2, 1.5, -0.1, 1.8],
        [1.5, 0.7,  0.3, 2.2],
        [0.9, 1.0,  0.2, 1.9]
    ])

    # Fake gradient coming from next layer (same shape as X) - different values
    d_out = np.array([
        [ 0.05, -0.02,  0.10,  0.20],
        [-0.10,  0.15, -0.05,  0.30],
        [ 0.12,  0.05,  0.02, -0.08],
        [ 0.00, -0.03,  0.07,  0.05],
        [ 0.09,  0.11, -0.02,  0.01]
    ])

    # Initialize BN for 4 features
    bn = BatchNorm(num_features=4, lr=0.01)
    out_bn = bn.forward(X, training=True)
    print("BatchNorm - Forward Output:\n", out_bn)

    dx_bn = bn.backward(d_out)
    print("\nBatchNorm - Grad wrt input (first sample):\n", dx_bn[0])
    print("BatchNorm - Updated gamma:\n", bn.gamma)
    print("BatchNorm - Updated beta:\n", bn.beta)

    # Initialize LN for 4 features
    ln = LayerNorm(num_features=4, lr=0.01)
    out_ln = ln.forward(X)
    print("\nLayerNorm - Forward Output:\n", out_ln)

    dx_ln = ln.backward(d_out)
    print("\nLayerNorm - Grad wrt input (all samples):\n", dx_ln)
    print("LayerNorm - Updated gamma:\n", ln.gamma)
    print("LayerNorm - Updated beta:\n", ln.beta)

if __name__ == "__main__":
    run_example()
