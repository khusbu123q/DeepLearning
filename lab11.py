import numpy as np

def pad_function(matrix, padding_size=1):
     rows, col = matrix.shape
     matrix = np.pad(matrix, ((padding_size, padding_size), (padding_size, padding_size)), mode="constant")
     return matrix

def convolution_op(input, filter, padding=1, stride=1):
    ip = pad_function(input, padding)
    input_shape = ip.shape[0]
    filter_shape = filter.shape[0]
    output_shape = ((input_shape - filter_shape) // stride) + 1
    output = []
    for i in range(0, output_shape * stride, stride):
        for j in range(0, output_shape * stride, stride):
            patch = ip[i:i + filter_shape, j:j + filter_shape]
            output.append((patch * filter).sum())
    output = np.array(output).reshape(output_shape, output_shape)
    return output

def max_pooling_op(input, filter_shape, stride=1):
    input_shape = input.shape[0]
    output_shape = ((input_shape - filter_shape) // stride) + 1
    output = []
    for i in range(0, output_shape * stride, stride):
        for j in range(0, output_shape * stride, stride):
            patch = input[i:i + filter_shape, j:j + filter_shape]
            output.append(patch.max())
    output = np.array(output).reshape(output_shape, output_shape)
    return output

if __name__ == "__main__":
    np.random.seed(0)
    image = np.random.randint(0, 255, size=(32, 32))
    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])

    conv_out = convolution_op(image, kernel, padding=1, stride=2)
    print("Convolution output shape:", conv_out.shape)
    print(conv_out)

    pool_out = max_pooling_op(conv_out, filter_shape=2, stride=2)
    print("Max pooling output shape:", pool_out.shape)
    print(pool_out)
