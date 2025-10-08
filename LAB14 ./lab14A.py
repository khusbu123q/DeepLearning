import numpy as np

# Set a random seed for reproducibility
np.random.seed(42)


class VanillaRNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, clip_grad=5.0):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.clip_grad = clip_grad

        # Small random initialization
        self.W_xh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_h = np.zeros((hidden_dim, 1))

        self.W_hy = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b_y = np.zeros((output_dim, 1))

    def forward(self, inputs):

        time_steps = inputs.shape[0]
        hidden_states = np.zeros((time_steps + 1, self.hidden_dim, 1))
        outputs = np.zeros((time_steps, self.output_dim, 1))

        for t in range(time_steps):
            x_t = inputs[t].reshape(-1, 1)  # (input_dim, 1)
            h_prev = hidden_states[t]      # (hidden_dim, 1)

            hidden_states[t + 1] = np.tanh(
                np.dot(self.W_xh, x_t) +
                np.dot(self.W_hh, h_prev) +
                self.b_h
            )

            outputs[t] = np.dot(self.W_hy, hidden_states[t + 1]) + self.b_y

        return hidden_states, outputs

    def backward(self, inputs, hidden_states, outputs, targets):

        time_steps = inputs.shape[0]

        # Gradients init
        d_Wxh = np.zeros_like(self.W_xh)
        d_Whh = np.zeros_like(self.W_hh)
        d_bh = np.zeros_like(self.b_h)
        d_Why = np.zeros_like(self.W_hy)
        d_by = np.zeros_like(self.b_y)

        d_next_h = np.zeros((self.hidden_dim, 1))

        for t in reversed(range(time_steps)):
            # Only compute output error for the final timestep (if that's the training objective)
            if t == time_steps - 1:
                dy = outputs[t] - targets[t].reshape(self.output_dim, 1)
            else:
                # No contribution to output loss from earlier timesteps
                dy = np.zeros((self.output_dim, 1))

            # Output layer gradients (only non-zero for final timestep)
            d_Why += np.dot(dy, hidden_states[t + 1].T)  # (output_dim, hidden_dim)
            d_by += dy

            # Backprop into hidden
            d_h = np.dot(self.W_hy.T, dy) + d_next_h     # (hidden_dim, 1)
            d_tanh = (1 - hidden_states[t + 1] ** 2)     # derivative of tanh
            d_h_raw = d_h * d_tanh

            x_t = inputs[t].reshape(-1, 1)
            d_Wxh += np.dot(d_h_raw, x_t.T)
            d_Whh += np.dot(d_h_raw, hidden_states[t].T)
            d_bh += d_h_raw

            d_next_h = np.dot(self.W_hh.T, d_h_raw)

        # Gradient clipping (element-wise)
        for g in [d_Wxh, d_Whh, d_bh, d_Why, d_by]:
            np.clip(g, -self.clip_grad, self.clip_grad, out=g)

        return d_Wxh, d_Whh, d_bh, d_Why, d_by

    def train(self, inputs, target):

        hidden_states, outputs = self.forward(inputs)

        # Compute loss on final timestep
        final_out = outputs[-1].reshape(self.output_dim, 1)
        target_vec = np.array(target).reshape(self.output_dim, 1)
        loss = np.mean((final_out - target_vec) ** 2)

        # build targets array where only final timestep has the target
        full_targets = np.zeros_like(outputs)
        full_targets[-1] = target_vec

        d_Wxh, d_Whh, d_bh, d_Why, d_by = self.backward(inputs, hidden_states, outputs, full_targets)

        # gradient descent update
        self.W_xh -= self.learning_rate * d_Wxh
        self.W_hh -= self.learning_rate * d_Whh
        self.b_h -= self.learning_rate * d_bh
        self.W_hy -= self.learning_rate * d_Why
        self.b_y -= self.learning_rate * d_by

        return loss

    def predict(self, inputs):

        _, outputs = self.forward(inputs)
        return outputs[-1].reshape(self.output_dim,)


def get_user_input():

    print("Please provide a sequence of numbers for the model to train on.")
    print("The model will learn to predict the next number in the sequence.")

    while True:
        try:
            sequence_length = int(input("Enter the length of the sequence: "))
            if sequence_length <= 0:
                print("Sequence length must be a positive integer.")
                continue

            print(f"Enter {sequence_length} space-separated numbers:")
            input_str = input()
            inputs = [float(x) for x in input_str.split()]

            if len(inputs) != sequence_length:
                print(f"You entered {len(inputs)} numbers, but the sequence length is {sequence_length}. Please try again.")
                continue

            target = float(input("Enter the target value (the next number in the sequence): "))
            break
        except ValueError:
            print("Invalid input. Please enter valid numbers.")

    return np.array(inputs), np.array([target]), sequence_length


if __name__ == "__main__":

    input_dim = 1
    hidden_dim = 10
    output_dim = 1
    epochs = 5000

    # Get user input
    user_inputs, user_targets, sequence_length = get_user_input()

    # Optional: normalize inputs (recommended if numbers are large / vary a lot)
    mean_in = user_inputs.mean()
    std_in = user_inputs.std() if user_inputs.std() > 0 else 1.0
    norm_inputs = (user_inputs - mean_in) / std_in

    # The same normalization must be applied to prediction inputs later.

    # reshape into (time_steps, input_dim)
    train_inputs = norm_inputs.reshape(sequence_length, input_dim)

    rnn = VanillaRNN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, learning_rate=0.005)

    print("\nStarting training...")

    for epoch in range(epochs):
        loss = rnn.train(train_inputs, user_targets)
        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.8f}")

    print("\nTraining finished!")

    # Prediction
    print("\nPlease provide a new sequence to get a prediction for.")
    print(f"It should be {sequence_length} numbers long (same normalization will be applied).")

    while True:
        try:
            prediction_input_str = input(f"Enter {sequence_length} space-separated numbers: ")
            prediction_input = np.array([float(x) for x in prediction_input_str.split()])
            if len(prediction_input) != sequence_length:
                print(f"The input must be {sequence_length} numbers long. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter valid numbers.")

    # Normalize using training mean/std
    norm_pred_input = (prediction_input - mean_in) / std_in
    pred_input = norm_pred_input.reshape(sequence_length, input_dim)

    pred = rnn.predict(pred_input)

    print("\nMaking a prediction...")
    print(f"Input sequence: {prediction_input}")
    print(f"Predicted next value: {pred[0]:.6f}")
