
from string import digits, punctuation
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


df = pd.read_csv("/home/ibab/Documents/Datasets/archive (5)/Hindi_English_Truncated_Corpus.csv", encoding="utf-8")
print(df["source"].value_counts())

df = df[df["source"] == "tides"].copy()
print(df.head(20))
print(pd.isnull(df).sum())

# drop rows without english/hindi
df = df[pd.notnull(df["english_sentence"]) & pd.notnull(df["hindi_sentence"])].copy()
df.drop_duplicates(inplace=True)

# sample for speed (you used 25k)
df = df.sample(n=25000, random_state=42).reset_index(drop=True)
print(df.shape)


def safe_lower(x):
    return str(x).lower()

df["english_sentence"] = df["english_sentence"].apply(safe_lower)
df["hindi_sentence"] = df["hindi_sentence"].apply(safe_lower)

# remove apostrophes
df['english_sentence'] = df['english_sentence'].str.replace("'", "", regex=False)
df['hindi_sentence'] = df['hindi_sentence'].str.replace("'", "", regex=False)

exclude = set(punctuation)
df["english_sentence"] = df["english_sentence"].apply(lambda s: "".join(ch for ch in s if ch not in exclude))
df["hindi_sentence"] = df["hindi_sentence"].apply(lambda s: "".join(ch for ch in s if ch not in exclude))

remove_digits = str.maketrans('', '', digits)
df['english_sentence'] = df['english_sentence'].apply(lambda x: x.translate(remove_digits))
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: x.translate(remove_digits))

# remove devanagari digits if present
df['hindi_sentence'] = df['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

# strip and collapse spaces
df['english_sentence'] = df['english_sentence'].str.strip().replace(r' +', ' ', regex=True)
df['hindi_sentence'] = df['hindi_sentence'].str.strip().replace(r' +', ' ', regex=True)


# Build vocab from the (filtered) df
all_eng_words = set()
for eng in df['english_sentence']:
    for w in eng.split():
        all_eng_words.add(w)

all_hin_words = set()
for hin in df['hindi_sentence']:
    for w in hin.split():
        all_hin_words.add(w)

print("Unique English words:", len(all_eng_words))
print("Unique Hindi words:", len(all_hin_words))

df['length_eng_sentence'] = df['english_sentence'].apply(lambda x: len(x.split()))
df['length_hin_sentence'] = df['hindi_sentence'].apply(lambda x: len(x.split()))

# filter long sentences
df = df[df["length_eng_sentence"] <= 20]
df = df[df["length_hin_sentence"] <= 20].reset_index(drop=True)
print(df.shape)

max_len_eng = df['length_eng_sentence'].max()
max_len_hin = df['length_hin_sentence'].max()
print("max English len:", max_len_eng, "max Hindi len:", max_len_hin)

# We'll use START_ and _END tokens for decoder
START_TOKEN = "START_"
END_TOKEN = "_END"
PAD_TOKEN = "<PAD>"

# Build token indices (start indices at 1; reserve 0 for padding)
input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_hin_words))

# Input (English) token -> index (1..N); 0 is padding
input_token_index = {w: i+1 for i, w in enumerate(input_words)}
# target (Hindi) token -> index
target_token_index = {w: i+1 for i, w in enumerate(target_words)}

# add special tokens for target
if START_TOKEN not in target_token_index:
    target_token_index[START_TOKEN] = len(target_token_index) + 1
if END_TOKEN not in target_token_index:
    target_token_index[END_TOKEN] = len(target_token_index) + 1

# note: padding will be index 0 implicitly

# reverse lookup
reverse_input_token_index = {i: w for w, i in input_token_index.items()}
reverse_target_token_index = {i: w for w, i in target_token_index.items()}

# vocabulary sizes (include index 0)
num_encoder_tokens = max(input_token_index.values()) + 1
num_decoder_tokens = max(target_token_index.values()) + 1

# encoder/decoder max lengths
max_encoder_seq_length = max_len_eng
max_decoder_seq_length = max_len_hin + 2  # for START_ and _END

print("num_encoder_tokens (including pad):", num_encoder_tokens)
print("num_decoder_tokens (including pad):", num_decoder_tokens)
print("max_encoder_seq_length:", max_encoder_seq_length)
print("max_decoder_seq_length:", max_decoder_seq_length)


def encode_encoder_sentence(sentence, token_index, max_length):
    tokens = [token_index.get(w, 0) for w in sentence.split()]  # unknown -> 0 (pad/unk)
    # pad to max_length
    tokens = tokens[:max_length] + [0] * (max_length - len(tokens))
    return tokens

def encode_decoder_sentence_with_start_end(sentence, token_index, max_length):
    # returns (dec_input, dec_target) both length = max_length
    words = sentence.split()
    token_ids = [token_index.get(w, 0) for w in words]
    # decoder input: START_ + tokens + padding (length = max_length)
    dec_in = [token_index[START_TOKEN]] + token_ids
    dec_in = dec_in[:max_length]
    dec_in = dec_in + [0] * (max_length - len(dec_in))
    # decoder target: tokens + END + padding (shifted)
    dec_target = token_ids + [token_index[END_TOKEN]]
    dec_target = dec_target[:max_length]
    dec_target = dec_target + [0] * (max_length - len(dec_target))
    return dec_in, dec_target


encoder_input_data = []
decoder_input_data = []
decoder_target_data = []

for eng, hin in zip(df['english_sentence'], df['hindi_sentence']):
    enc = encode_encoder_sentence(eng, input_token_index, max_encoder_seq_length)
    dec_in, dec_tgt = encode_decoder_sentence_with_start_end(hin, target_token_index, max_decoder_seq_length)
    encoder_input_data.append(enc)
    decoder_input_data.append(dec_in)
    decoder_target_data.append(dec_tgt)

encoder_input_data = torch.tensor(np.array(encoder_input_data), dtype=torch.long)
decoder_input_data = torch.tensor(np.array(decoder_input_data), dtype=torch.long)
decoder_target_data = torch.tensor(np.array(decoder_target_data), dtype=torch.long)

print("Shapes:", encoder_input_data.shape, decoder_input_data.shape, decoder_target_data.shape)


(enc_train, enc_val,
 dec_in_train, dec_in_val,
 dec_tgt_train, dec_tgt_val,
 eng_sent_train, eng_sent_val,
 hin_sent_train, hin_sent_val) = train_test_split(
    encoder_input_data, decoder_input_data, decoder_target_data,
    df['english_sentence'].values, df['hindi_sentence'].values,
    test_size=0.2, random_state=42)

batch_size = 64
train_dataset = TensorDataset(enc_train, dec_in_train, dec_tgt_train)
val_dataset = TensorDataset(enc_val, dec_in_val, dec_tgt_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size, padding_idx=0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_size, hidden_size, padding_idx=0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(target_vocab_size, embed_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, target_vocab_size)

    def forward(self, x, hidden, cell):
        # x: (batch, seq_len) where seq_len often = 1 during inference, or >1 during teacher forcing
        x = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        outputs = self.fc(outputs)  # (batch, seq_len, vocab)
        return outputs, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg):
        # src: (batch, src_len)
        # trg: (batch, trg_len) decoder input (with START_)
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        decoder_input = trg[:, :1]  # first token is START_
        for t in range(1, trg_len):
            out, hidden, cell = self.decoder(decoder_input, hidden, cell)  # out: (batch, seq_len=1, vocab)
            outputs[:, t, :] = out[:, 0, :]
            # teacher forcing: next input is actual trg token at t
            decoder_input = trg[:, t].unsqueeze(1)

        return outputs


embedding_dim = 256
hidden_size = 512
encoder = Encoder(num_encoder_tokens, embedding_dim, hidden_size, padding_idx=0).to(device)
decoder = Decoder(num_decoder_tokens, embedding_dim, hidden_size, padding_idx=0).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for enc_seq, dec_seq_in, dec_seq_tgt in train_loader:
        enc_seq = enc_seq.to(device)
        dec_seq_in = dec_seq_in.to(device)
        dec_seq_tgt = dec_seq_tgt.to(device)

        optimizer.zero_grad()
        output = model(enc_seq, dec_seq_in)  # (batch, trg_len, vocab)
        # remove the first step (corresponding to t=0 placeholder) to align
        output = output[:, 1:, :].contiguous()
        dec_target = dec_seq_tgt[:, :output.shape[1]].contiguous()

        batch_size_, seq_len_, vocab_size_ = output.shape
        output = output.view(batch_size_ * seq_len_, vocab_size_)
        dec_target = dec_target.view(-1)

        loss = criterion(output, dec_target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for enc_seq, dec_seq_in, dec_seq_tgt in val_loader:
            enc_seq = enc_seq.to(device)
            dec_seq_in = dec_seq_in.to(device)
            dec_seq_tgt = dec_seq_tgt.to(device)

            output = model(enc_seq, dec_seq_in)
            output = output[:, 1:, :].contiguous()
            dec_target = dec_seq_tgt[:, :output.shape[1]].contiguous()

            batch_size_, seq_len_, vocab_size_ = output.shape
            output = output.view(batch_size_ * seq_len_, vocab_size_)
            dec_target = dec_target.view(-1)
            loss = criterion(output, dec_target)
            val_loss += loss.item()

        val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")


def translate_sentence(sentence, max_encoder_len=max_encoder_seq_length, max_decoder_len=max_decoder_seq_length):
    model.eval()
    # encode source
    enc_tokens = encode_encoder_sentence(sentence, input_token_index, max_encoder_len)
    enc_tensor = torch.tensor([enc_tokens], dtype=torch.long).to(device)
    hidden, cell = encoder(enc_tensor)


    decoder_input = torch.tensor([[target_token_index[START_TOKEN]]], dtype=torch.long).to(device)
    decoded_words = []

    for _ in range(max_decoder_len):
        out, hidden, cell = decoder(decoder_input, hidden, cell)  # out shape (1, seq_len, vocab)
        pred_id = out.argmax(2)[:, -1].item()
        # if padding or unknown (0) predicted, break
        if pred_id == 0:
            break
        word = reverse_target_token_index.get(pred_id, "")
        if word == END_TOKEN:
            break
        decoded_words.append(word)
        decoder_input = torch.tensor([[pred_id]], dtype=torch.long).to(device)

    return " ".join(decoded_words)


sample_idx = 0
sample_eng = eng_sent_val[sample_idx]
sample_hin = hin_sent_val[sample_idx]
pred_hin = translate_sentence(sample_eng)

print("English:", sample_eng)
print("Actual Hindi:", sample_hin)
print("Predicted Hindi:", pred_hin)
