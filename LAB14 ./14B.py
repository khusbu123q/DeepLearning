import numpy as np

np.random.seed(42)


class WordRNN:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, learning_rate=0.01, clip_grad=5.0):
        """
        Word-level vanilla RNN that predicts the next word (classification over vocab).
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.clip_grad = clip_grad

        # Embedding matrix: shape (embedding_dim, vocab_size)
        self.Emb = np.random.randn(embedding_dim, vocab_size) * 0.01

        # RNN weights
        self.W_xh = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b_h = np.zeros((hidden_dim, 1))

        # Output layer: logits over vocabulary
        self.W_hy = np.random.randn(vocab_size, hidden_dim) * 0.01
        self.b_y = np.zeros((vocab_size, 1))

    def forward(self, input_indices):
        """
        input_indices: list/array of token indices (length = time_steps)
        returns: hidden_states (time_steps+1, hidden_dim,1), logits (time_steps, vocab_size,1), probs (time_steps, vocab_size,1)
        """
        time_steps = len(input_indices)
        hidden_states = np.zeros((time_steps + 1, self.hidden_dim, 1))
        logits = np.zeros((time_steps, self.vocab_size, 1))
        probs = np.zeros_like(logits)

        for t in range(time_steps):
            idx = input_indices[t]
            x_t = self.Emb[:, idx].reshape(-1, 1)        # (embedding_dim,1)
            h_prev = hidden_states[t]                   # (hidden_dim,1)

            hidden_states[t + 1] = np.tanh(
                np.dot(self.W_xh, x_t) + np.dot(self.W_hh, h_prev) + self.b_h
            )

            logits[t] = np.dot(self.W_hy, hidden_states[t + 1]) + self.b_y  # (vocab_size,1)

            # softmax (numerically stable)
            z = logits[t]
            z = z - np.max(z)
            exp_z = np.exp(z)
            probs[t] = exp_z / np.sum(exp_z)

        return hidden_states, logits, probs

    def backward(self, input_indices, hidden_states, logits, probs, target_onehot):
        """
        Backpropagation through time.
        target_onehot: (time_steps, vocab_size, 1) with non-zero only at final timestep (we predict last)
        """
        time_steps = len(input_indices)

        # init grads
        dEmb = np.zeros_like(self.Emb)       # (embedding_dim, vocab_size)
        d_Wxh = np.zeros_like(self.W_xh)
        d_Whh = np.zeros_like(self.W_hh)
        d_bh = np.zeros_like(self.b_h)
        d_Why = np.zeros_like(self.W_hy)
        d_by = np.zeros_like(self.b_y)

        d_next_h = np.zeros((self.hidden_dim, 1))

        for t in reversed(range(time_steps)):
            # softmax derivative combined with cross-entropy:
            # Only final timestep has target; other timesteps contribute zero.
            if np.any(target_onehot[t]):
                dy = probs[t] - target_onehot[t]   # (vocab_size,1)
            else:
                dy = np.zeros((self.vocab_size, 1))

            # Output layer grads
            d_Why += np.dot(dy, hidden_states[t + 1].T)   # (vocab_size, hidden_dim)
            d_by += dy

            # backprop into hidden
            d_h = np.dot(self.W_hy.T, dy) + d_next_h       # (hidden_dim,1)
            d_tanh = (1 - hidden_states[t + 1] ** 2)
            d_h_raw = d_h * d_tanh                         # (hidden_dim,1)

            # embedding index
            idx = input_indices[t]
            x_t = self.Emb[:, idx].reshape(-1, 1)          # (embedding_dim,1)

            d_Wxh += np.dot(d_h_raw, x_t.T)                # (hidden_dim, embedding_dim)
            d_Whh += np.dot(d_h_raw, hidden_states[t].T)   # (hidden_dim, hidden_dim)
            d_bh += d_h_raw

            # gradient w.r.t. embedding vector for this token
            d_emb_vec = np.dot(self.W_xh.T, d_h_raw).reshape(-1)  # (embedding_dim,)
            dEmb[:, idx] += d_emb_vec

            # carry to previous
            d_next_h = np.dot(self.W_hh.T, d_h_raw)

        # clip grads
        for g in [dEmb, d_Wxh, d_Whh, d_bh, d_Why, d_by]:
            np.clip(g, -self.clip_grad, self.clip_grad, out=g)

        return dEmb, d_Wxh, d_Whh, d_bh, d_Why, d_by

    def train_on_example(self, input_indices, target_index):
        """
        Performs forward, computes loss (cross-entropy on final timestep), backward and update.
        """
        hidden_states, logits, probs = self.forward(input_indices)
        time_steps = len(input_indices)

        # Loss: cross-entropy on final timestep
        p_final = probs[-1]  # (vocab_size,1)
        loss = -np.log(p_final[target_index, 0] + 1e-12)

        # build target_onehot array shaped like probs
        target_onehot = np.zeros_like(probs)
        target_onehot[-1, target_index, 0] = 1.0

        dEmb, d_Wxh, d_Whh, d_bh, d_Why, d_by = self.backward(input_indices, hidden_states, logits, probs, target_onehot)

        # update parameters
        self.Emb -= self.learning_rate * dEmb
        self.W_xh -= self.learning_rate * d_Wxh
        self.W_hh -= self.learning_rate * d_Whh
        self.b_h -= self.learning_rate * d_bh
        self.W_hy -= self.learning_rate * d_Why
        self.b_y -= self.learning_rate * d_by

        return loss

    def predict_next(self, input_indices):
        """
        Predicts the next word index (argmax) and returns probability distribution.
        """
        _, _, probs = self.forward(input_indices)
        p_final = probs[-1].reshape(-1)
        pred_idx = int(np.argmax(p_final))
        return pred_idx, p_final



def tokenize(text):
    # simple whitespace tokenizer; lowercasing
    return text.strip().lower().split()


def build_vocab(sentences, add_unk=True):
    """
    sentences: list of strings
    returns: word2idx, idx2word
    """
    tokens = []
    for s in sentences:
        tokens.extend(tokenize(s))
    vocab = sorted(list(dict.fromkeys(tokens)))  # preserve order of first appearance
    word2idx = {}
    idx2word = {}
    next_idx = 0
    if add_unk:
        word2idx["<UNK>"] = next_idx
        idx2word[next_idx] = "<UNK>"
        next_idx += 1
    for w in vocab:
        if w not in word2idx:
            word2idx[w] = next_idx
            idx2word[next_idx] = w
            next_idx += 1
    return word2idx, idx2word


def words_to_indices(words, word2idx):
    return [word2idx[w] if w in word2idx else word2idx.get("<UNK>", 0) for w in words]



if __name__ == "__main__":
    print("Enter a training sentence (this will build the vocabulary).")
    print("Example: 'the cat sat on the mat' and target: 'on' (next word).")
    sentence = input("Training sentence: ").strip()
    if len(sentence) == 0:
        raise SystemExit("No sentence provided. Exiting.")

    # build vocab from the single sentence (you can later add more sentences)
    word2idx, idx2word = build_vocab([sentence], add_unk=True)
    vocab_size = len(word2idx)
    print(f"Vocabulary ({vocab_size} tokens): {word2idx}")

    # get sequence length and inputs
    tokens = tokenize(sentence)
    while True:
        try:
            seq_len = int(input(f"Enter how many words from the start to use as input (1..{len(tokens)}): "))
            if 1 <= seq_len <= len(tokens):
                break
            else:
                print("Invalid number.")
        except ValueError:
            print("Enter an integer.")

    input_words = tokens[:seq_len]
    print("Input words:", input_words)

    # ask target next word
    target_word = input("Enter target next word (the model will learn to predict this word): ").strip().lower()
    if target_word not in word2idx:
        print(f"Target word '{target_word}' not in vocab — it will be treated as <UNK>.")
    input_indices = words_to_indices(input_words, word2idx)
    target_index = word2idx.get(target_word, word2idx.get("<UNK>", 0))

    # hyperparams
    embedding_dim = 16
    hidden_dim = 32
    epochs = 2000
    lr = 0.01

    rnn = WordRNN(vocab_size=vocab_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, learning_rate=lr)
    print("\nStarting training (predict next word)...")
    for ep in range(1, epochs + 1):
        loss = rnn.train_on_example(input_indices, target_index)
        if ep % 200 == 0:
            print(f"Epoch {ep}/{epochs}, Loss: {loss:.4f}")

    print("Training finished.\n")

    # interactive prediction on new input
    print("Now you can give a new sequence (same or shorter) to predict the next word.")
    while True:
        new_input = input(f"Enter up to {seq_len} words (or 'quit' to exit): ").strip()
        if new_input.lower() in ("quit", "exit"):
            break
        new_tokens = tokenize(new_input)
        if len(new_tokens) == 0:
            print("No input, try again.")
            continue
        if len(new_tokens) > seq_len:
            new_tokens = new_tokens[:seq_len]
            print(f"Truncated to first {seq_len} tokens: {new_tokens}")

        new_indices = words_to_indices(new_tokens, word2idx)

        pred_idx, probs = rnn.predict_next(new_indices)
        pred_word = idx2word[pred_idx]
        print(f"Predicted next word: '{pred_word}' (prob {probs[pred_idx]:.4f})")
        # optionally show top 5 probabilities
        top_k = 5
        top_indices = np.argsort(probs)[-top_k:][::-1]
        print("Top predictions:")
        for i in top_indices:
            print(f"  {idx2word[int(i)]}: {probs[int(i)]:.4f}")
        print()
