# image_captioning_pytorch.py
import os
import re
import string
import random
import pickle
from collections import Counter

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


data_location = "/home/ibab/khusbu_pycharm/pythonProject 1/class/DEEPLEARNING/dataset"
images_path = os.path.join(data_location, "Images")
captions_file = os.path.join(data_location, "captions.txt")

MAX_CAPTION_LENGTH = 34
VOCAB_SIZE = 8000   # max vocab size (including special tokens)
EMBEDDING_DIM = 256
UNITS = 512
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_text_descriptions(filename, verbose=False):
    descriptions = {}
    bad_lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            if '\t' in line:
                parts = line.split('\t', 1)
            elif '#' in line:
                after_hash = line.split('#', 1)
                if len(after_hash) == 2 and ' ' in after_hash[1]:
                    left = after_hash[0] + '#' + after_hash[1].split(' ', 1)[0]
                    right = after_hash[1].split(' ', 1)[1]
                    parts = [left, right]
                else:
                    parts = line.split(' ', 1)
            elif ' ' in line:
                parts = line.split(' ', 1)
            else:
                parts = [line]

            if len(parts) < 2 or not parts[1].strip():
                bad_lines.append((lineno, line))
                continue

            image_id_part = parts[0].strip()
            caption = parts[1].strip()

            if '#' in image_id_part:
                image_id_part = image_id_part.split('#', 1)[0]
            image_id = os.path.splitext(image_id_part)[0]

            descriptions.setdefault(image_id, []).append(caption)

    if verbose:
        print(f"Loaded descriptions for {len(descriptions)} images.")
        if bad_lines:
            print(f"Skipped {len(bad_lines)} malformed/empty lines. Example(s):")
            for ln, ln_txt in bad_lines[:5]:
                print(f"  line {ln}: {ln_txt}")

    return descriptions


def clean_captions(descriptions):
    table = str.maketrans('', '', string.punctuation)
    for image_id, caption_list in descriptions.items():
        for i, caption in enumerate(caption_list):
            caption = caption.lower()
            caption = caption.translate(table)
            caption = re.sub(r'\d+', '', caption)
            caption = ' '.join([w for w in caption.split() if len(w) > 1])
            caption = 'startseq ' + caption + ' endseq'
            descriptions[image_id][i] = caption

def all_captions_list(descriptions):
    all_caps = []
    for k in descriptions:
        all_caps.extend(descriptions[k])
    return all_caps

class SimpleTokenizer:
    def __init__(self, vocab_size=VOCAB_SIZE, oov_token='<unk>'):
        self.vocab_size = vocab_size
        self.oov_token = oov_token
        self.word2idx = {}
        self.idx2word = {}
        self.pad_token = '<pad>'
        self.start_token = 'startseq'
        self.end_token = 'endseq'

    def fit_on_texts(self, texts):
        counter = Counter()
        for t in texts:
            for w in t.split():
                counter[w] += 1
        most_common = counter.most_common(self.vocab_size - 4)  # reserve tokens
        self.word2idx = {self.pad_token:0, self.start_token:1, self.end_token:2, self.oov_token:3}
        idx = 4
        for w, _ in most_common:
            if w in self.word2idx:
                continue
            self.word2idx[w] = idx
            idx += 1
        # build idx2word mapping (index -> word)
        self.idx2word = {i:w for w,i in self.word2idx.items()}

    def texts_to_sequences(self, texts):
        seqs = []
        for t in texts:
            seq = []
            for w in t.split():
                seq.append(self.word2idx.get(w, self.word2idx[self.oov_token]))
            seqs.append(seq)
        return seqs

    def sequence_to_text(self, seq):
        words = []
        for idx in seq:
            if idx == 0: # pad
                continue
            words.append(self.idx2word.get(idx, self.oov_token))
        return ' '.join(words)

    def vocab_size_actual(self):
        return len(self.word2idx)


def build_vgg_feature_extractor(device=DEVICE):
    # note: torchvision warns about `pretrained` deprecation; this still works
    vgg = models.vgg16(pretrained=True)
    for p in vgg.parameters():
        p.requires_grad = False
    features = nn.Sequential(
        vgg.features,
        nn.AdaptiveAvgPool2d((7, 7)),
        nn.Flatten(),
        vgg.classifier[:-1]  # remove final 1000-class layer -> get 4096-d vector
    )
    features.to(device)
    features.eval()
    return features

def extract_features_from_images(images_dir, model, device=DEVICE):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    features = {}
    for img_name in tqdm(os.listdir(images_dir), desc="Extracting image features"):
        img_path = os.path.join(images_dir, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(x)             # shape (1, 4096)
            img_id = img_name.split('.')[0]
            features[img_id] = feat.cpu().squeeze(0).numpy()
        except Exception as e:
            print(f"Skipping {img_name}: {e}")
            continue
    return features


def pad_sequence(seq, max_len, pad_value=0):
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + [pad_value] * (max_len - len(seq))

class CaptionDataset(Dataset):
    def __init__(self, image_caption_pairs, features_dict, tokenizer, max_length=MAX_CAPTION_LENGTH):
        self.pairs = [p for p in image_caption_pairs if p[0] in features_dict]
        self.features = features_dict
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_idx = self.tokenizer.word2idx[self.tokenizer.pad_token]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_id, caption = self.pairs[idx]
        img_feat = torch.tensor(self.features[img_id], dtype=torch.float32)  # (4096,)
        seq = self.tokenizer.texts_to_sequences([caption])[0]
        seq_padded = pad_sequence(seq, self.max_length, pad_value=self.pad_idx)
        seq_tensor = torch.tensor(seq_padded, dtype=torch.long)
        return img_feat, seq_tensor

def collate_fn(batch):
    img_feats = torch.stack([item[0] for item in batch], dim=0)   # (B, 4096)
    seqs = torch.stack([item[1] for item in batch], dim=0)       # (B, max_len)
    inputs = seqs[:, :-1]
    targets = seqs[:, 1:]
    return img_feats, inputs, targets

class DecoderWithInit(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, img_feat_dim=4096, padding_idx=0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.img2h = nn.Linear(img_feat_dim, hidden_dim)
        self.img2c = nn.Linear(img_feat_dim, hidden_dim)
        self.img_proj = nn.Linear(img_feat_dim, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim*2, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, img_feats, captions):
        emb = self.embed(captions)                     # (B, seq_len, embed_dim)
        img_emb = self.img_proj(img_feats).unsqueeze(1)  # (B, 1, embed_dim)
        img_emb_rep = img_emb.repeat(1, emb.size(1), 1)
        lstm_input = torch.cat([emb, img_emb_rep], dim=2)

        h0 = torch.tanh(self.img2h(img_feats)).unsqueeze(0)
        c0 = torch.tanh(self.img2c(img_feats)).unsqueeze(0)
        out, _ = self.lstm(lstm_input, (h0, c0))
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

    def generate(self, img_feat, tokenizer, max_length=MAX_CAPTION_LENGTH, device=DEVICE):
     
        self.eval()
        idx2word = tokenizer.idx2word
        word2idx = tokenizer.word2idx
        pad_idx = word2idx[tokenizer.pad_token]

        # Convert img_feat (numpy array or 1D tensor) -> tensor with batch dim (1, img_feat_dim)
        if isinstance(img_feat, np.ndarray):
            img_tensor = torch.tensor(img_feat, dtype=torch.float32).unsqueeze(0).to(device)
        elif torch.is_tensor(img_feat):
            img_tensor = img_feat.to(device)
            if img_tensor.dim() == 1:
                img_tensor = img_tensor.unsqueeze(0)
        else:
            img_tensor = torch.tensor(np.array(img_feat), dtype=torch.float32).unsqueeze(0).to(device)

        # Project image -> embedding and initial hidden/cell
        img_emb = self.img_proj(img_tensor).unsqueeze(1)  # (1, 1, embed_dim)
        h = torch.tanh(self.img2h(img_tensor)).unsqueeze(0)  # (1, 1, hidden_dim)
        c = torch.tanh(self.img2c(img_tensor)).unsqueeze(0)  # (1, 1, hidden_dim)

        generated = [word2idx[tokenizer.start_token]]

        for t in range(max_length):
            cur_input = torch.tensor([generated[-1]], dtype=torch.long).unsqueeze(0).to(device)  # (1,1)
            emb = self.embed(cur_input)  # (1,1,embed_dim)
            lstm_in = torch.cat([emb, img_emb], dim=2)  # (1,1,2*embed_dim)
            out, (h, c) = self.lstm(lstm_in, (h, c))  # out: (1,1,hidden_dim)
            logits = self.fc(out.squeeze(1))  # (1, vocab)
            next_word = torch.argmax(logits, dim=1).item()
            generated.append(next_word)
            if next_word == word2idx[tokenizer.end_token]:
                break

        # Convert indices -> words, skip start/end/pad
        words = []
        for idx in generated:
            if idx == word2idx[tokenizer.start_token] or idx == word2idx[tokenizer.end_token] or idx == pad_idx:
                continue
            words.append(idx2word.get(idx, tokenizer.oov_token))
        return ' '.join(words)


def train_model(model, dataloader, val_loader, epochs, lr, device, tokenizer):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx[tokenizer.pad_token])
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        for img_feats, inputs, targets in tqdm(dataloader, desc=f"Train Epoch {epoch}"):
            img_feats = img_feats.to(device)
            inputs = inputs.to(device)
            targets = targets.to(device)   # (B, seq_len)

            optimizer.zero_grad()
            logits = model(img_feats, inputs)  # (B, seq_len, vocab)

            # ensure contiguous before view (fixes the runtime error)
            logits_flat = logits.contiguous().view(-1, logits.size(-1))   # (B*seq_len, vocab)
            targets_flat = targets.contiguous().view(-1)                 # (B*seq_len,)

            loss = criterion(logits_flat, targets_flat)
            loss.backward()

            # optional: gradient clipping to stabilize training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch} Train loss: {avg_loss:.4f}")

        # validation loop
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img_feats, inputs, targets in val_loader:
                    img_feats = img_feats.to(device)
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    logits = model(img_feats, inputs)
                    logits_flat = logits.contiguous().view(-1, logits.size(-1))
                    targets_flat = targets.contiguous().view(-1)

                    loss = criterion(logits_flat, targets_flat)
                    val_loss += loss.item()

            val_avg = val_loss / len(val_loader)
            print(f"Epoch {epoch} Val loss: {val_avg:.4f}")

    return model


def main():
    descriptions = load_text_descriptions(captions_file)
    clean_captions(descriptions)
    print(f"Loaded captions for {len(descriptions)} images.")

    all_ids = [n.split('.')[0] for n in os.listdir(images_path)]
    descriptions = {k:v for k,v in descriptions.items() if k in all_ids}
    all_caps = all_captions_list(descriptions)
    tokenizer = SimpleTokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.fit_on_texts(all_caps)
    vocab_size = tokenizer.vocab_size_actual()
    print(f"Vocab size (actual): {vocab_size}")

    vgg_model = build_vgg_feature_extractor(device=DEVICE)
    features_file = os.path.join(data_location, 'image_features.pkl')
    if os.path.exists(features_file):
        print("Loading cached image features...")
        with open(features_file, 'rb') as f:
            image_features = pickle.load(f)
    else:
        image_features = extract_features_from_images(images_path, vgg_model, device=DEVICE)
        with open(features_file, 'wb') as f:
            pickle.dump(image_features, f)
    print(f"Extracted features for {len(image_features)} images.")

    pairs = []
    for img_id in image_features.keys():
        caps = descriptions.get(img_id, [])
        for c in caps:
            pairs.append((img_id, c))

    random.seed(42)
    random.shuffle(pairs)
    split = int(0.8 * len(pairs))
    train_pairs = pairs[:split]
    val_pairs = pairs[split:]
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    train_ds = CaptionDataset(train_pairs, image_features, tokenizer, max_length=MAX_CAPTION_LENGTH)
    val_ds = CaptionDataset(val_pairs, image_features, tokenizer, max_length=MAX_CAPTION_LENGTH)

    # DataLoader: adjust num_workers/pin_memory if you use CUDA and have many CPUs
    num_workers = 4 if os.cpu_count() and os.cpu_count() >= 4 else 0
    pin_memory = True if DEVICE.type == 'cuda' else False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=num_workers, pin_memory=pin_memory)

    model = DecoderWithInit(embed_dim=EMBEDDING_DIM, hidden_dim=UNITS, vocab_size=vocab_size, img_feat_dim=4096, padding_idx=tokenizer.word2idx[tokenizer.pad_token])

    model = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE, tokenizer=tokenizer)

    torch.save(model.state_dict(), 'image_captioning_decoder.pth')
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    print("Saved model and tokenizer.")

    test_img_id = list(image_features.keys())[0]
    feat = image_features[test_img_id]
    caption = model.generate(feat, tokenizer, max_length=MAX_CAPTION_LENGTH, device=DEVICE)
    print(f"Generated caption for {test_img_id}.jpg : {caption}")

if __name__ == "__main__":
    main()
