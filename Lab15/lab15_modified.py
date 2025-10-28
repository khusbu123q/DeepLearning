

import os
import re
import math
import json
from collections import Counter
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split


DATA_DIR = "/home/ibab/khusbu_pycharm/pythonProject 1/class/DEEPLEARNING/dataset"
IMAGES_DIR = os.path.join(DATA_DIR, "Images")
CAPTIONS_PATH = os.path.join(DATA_DIR, "captions.txt")   # CSV with columns: image_name,caption_text

BATCH_SIZE = 32
EMBED_DIM = 256
NUM_DECODER_LAYERS = 3
NUM_HEADS = 4
DIM_FF = 1024
NUM_EPOCHS = 3
NUM_WORKERS = 4
FREQ_THRESHOLD = 5   # vocabulary frequency threshold
MAX_GEN_LEN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 50
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


class TextVocab:
    def __init__(self, freq_threshold: int = 5):
        self.freq_threshold = freq_threshold
        # reserved tokens
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def build_vocab(self, sentences: List[str]):
        """
        Build vocabulary by counting tokens and adding tokens when they reach freq_threshold.
        (This mirrors your original logic.)
        """
        freq = Counter()
        next_idx = max(self.itos.keys()) + 1  # start at 4
        for s in sentences:
            for w in simple_tokenize(s):
                freq[w] += 1
                # add token the moment it reaches threshold
                if freq[w] == self.freq_threshold and w not in self.stoi:
                    self.stoi[w] = next_idx
                    self.itos[next_idx] = w
                    next_idx += 1

    def numericalize(self, text: str) -> List[int]:
        return [self.stoi.get(t, self.stoi["<UNK>"]) for t in simple_tokenize(text)]

    def decode_ids(self, ids: List[int]) -> List[str]:
        return [self.itos.get(i, "<UNK>") for i in ids]

# -------------------------
# Dataset & Collate
# -------------------------
class CaptionDataset(Dataset):
    def __init__(self, images_root: str, captions_df: pd.DataFrame, vocab: TextVocab = None,
                 transform=None, freq_threshold: int = 5):
        self.root = images_root
        self.df = captions_df.reset_index(drop=True)
        self.transform = transform
        self.img_names = self.df["image_name"].tolist()
        self.captions = self.df["caption_text"].tolist()

        if vocab is None:
            self.vocab = TextVocab(freq_threshold=freq_threshold)
            self.vocab.build_vocab(self.captions)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        caption = self.captions[idx]
        img_path = os.path.join(self.root, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        token_ids = [self.vocab.stoi["<SOS>"]]
        token_ids += self.vocab.numericalize(caption)
        token_ids.append(self.vocab.stoi["<EOS>"])
        return image, torch.tensor(token_ids, dtype=torch.long)

class PadCollate:
    def __init__(self, padding_idx: int):
        self.padding_idx = padding_idx

    def __call__(self, batch):
        images = [x[0].unsqueeze(0) for x in batch]
        images = torch.cat(images, dim=0)
        captions = [x[1] for x in batch]
        captions_padded = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=self.padding_idx)
        return images, captions_padded

# -------------------------
# Transforms
# -------------------------
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))
])

# -------------------------
# Positional Encoding
# -------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(x.device)

# -------------------------
# Encoder: ResNet -> patch features
# -------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # remove avgpool and fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.proj = nn.Linear(512, embedding_dim)
        self.ln = nn.LayerNorm(embedding_dim)
        # freeze backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # returns (B, S, E) where S = 7*7
        x = self.backbone(images)        # (B, 512, H', W')
        x = self.adaptive_pool(x)        # (B, 512, 7, 7)
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # (B, S, C)
        x = self.proj(x)                # (B, S, E)
        x = self.ln(x)
        return x

# -------------------------
# Transformer decoder & create masks
# -------------------------
def create_tgt_masks(tgt_tokens: torch.Tensor, pad_idx: int):
    # tgt_tokens: (B, T)
    B, T = tgt_tokens.shape
    key_padding_mask = (tgt_tokens == pad_idx)      # (B, T) True for pad
    tgt = tgt_tokens.transpose(0, 1).contiguous()    # (T, B)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_tokens.device)  # (T, T)
    return tgt, tgt_mask, key_padding_mask

class TransformerCaptionDecoder(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, nhead: int = 4,
                 num_layers: int = 3, dim_feedforward: int = 1024, dropout: float = 0.1, max_len: int = 64):
        super().__init__()
        self.d_model = embedding_dim
        self.token_embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_enc = PositionalEncoding(embedding_dim, max_len=max_len)
        dec_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead,
                                               dim_feedforward=dim_feedforward, dropout=dropout, activation='relu')
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tgt_tokens: torch.Tensor, memory: torch.Tensor,
                tgt_mask=None, tgt_key_padding_mask=None) -> torch.Tensor:
        # tgt_tokens: (T, B)
        emb = self.token_embed(tgt_tokens) * math.sqrt(self.d_model)  # (T, B, E)
        # convert to (B, T, E) for PosEnc and back -> reusing PosEnc design
        emb = self.pos_enc(emb.transpose(0,1)).transpose(0,1)        # (T, B, E)
        if tgt_mask is None:
            T = tgt_tokens.size(0)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(T).to(tgt_tokens.device)
        out = self.transformer_decoder(tgt=emb, memory=memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        logits = self.out(out)  # (T, B, V)
        return logits

    @torch.no_grad()
    def generate(self, memory: torch.Tensor, vocab: TextVocab, max_length: int = 20, device: str = 'cpu'):
        # memory: (S, 1, E)
        generated = [vocab.stoi["<SOS>"]]
        for _ in range(max_length):
            tgt_tensor = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(1)  # (T,1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_tensor.size(0)).to(device)
            logits = self.forward(tgt_tensor, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=None)  # (T,1,V)
            next_logits = logits[-1, 0, :]  # (V,)
            next_id = int(next_logits.argmax().cpu().numpy())
            generated.append(next_id)
            if next_id == vocab.stoi["<EOS>"]:
                break
        tokens = [vocab.itos.get(i, "<UNK>") for i in generated[1:]]
        if tokens and tokens[-1] == "<EOS>":
            tokens = tokens[:-1]
        return tokens

# -------------------------
# Full model combining encoder & decoder
# -------------------------
class ImageCaptionNet(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int, nhead: int, num_layers: int):
        super().__init__()
        self.encoder = FeatureExtractor(embedding_dim)
        self.decoder = TransformerCaptionDecoder(embedding_dim, vocab_size, nhead=nhead, num_layers=num_layers)

    def forward(self, images: torch.Tensor, captions: torch.Tensor, pad_idx: int):
        # images: (B,3,H,W); captions: (B,T)
        mem = self.encoder(images)                  # (B, S, E)
        mem = mem.permute(1, 0, 2).contiguous()     # (S, B, E)
        tgt, tgt_mask, key_padding_mask = create_tgt_masks(captions, pad_idx=pad_idx)
        logits = self.decoder(tgt, mem, tgt_mask=tgt_mask, tgt_key_padding_mask=key_padding_mask)  # (T,B,V)
        logits = logits.transpose(0,1).contiguous()  # (B,T,V)
        return logits

    def generate_caption(self, image: torch.Tensor, vocab: TextVocab, max_length: int = 20, device: str = 'cpu'):
        # image: (1,3,H,W)
        mem = self.encoder(image.to(device))          # (1, S, E)
        mem = mem.permute(1, 0, 2).contiguous()       # (S,1,E)
        tokens = self.decoder.generate(mem, vocab, max_length=max_length, device=device)
        return tokens

# -------------------------
# BLEU scorer (simple, same as earlier)
# -------------------------
def compute_bleu_score(references: List[List[str]], candidate: List[str]) -> float:
    from collections import Counter
    import numpy as np
    def ngram_counter(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    def modified_prec(refs, cand, n):
        cand_counts = ngram_counter(cand, n)
        if not cand_counts:
            return 0.0
        max_counts = Counter()
        for ref in refs:
            ref_counts = ngram_counter(ref, n)
            for ng in ref_counts:
                max_counts[ng] = max(max_counts[ng], ref_counts[ng])
        clipped = {ng: min(count, max_counts.get(ng,0)) for ng,count in cand_counts.items()}
        return sum(clipped.values()) / sum(cand_counts.values())
    weights = [0.25] * 4
    precisions = [modified_prec(references, candidate, i+1) for i in range(4)]
    precisions = [p if p > 0 else 1e-9 for p in precisions]
    bleu = float(np.exp(sum(w * math.log(p) for w,p in zip(weights, precisions))))
    return bleu

# -------------------------
# MAIN: data, model, training, eval
# -------------------------
def main():
    # -- Load captions CSV
    if not os.path.exists(CAPTIONS_PATH):
        raise FileNotFoundError(f"Captions file not found: {CAPTIONS_PATH}")
    df = pd.read_csv(CAPTIONS_PATH, delimiter=",")
    # normalize columns
    if list(df.columns)[:2] != ["image_name", "caption_text"]:
        df.columns = ["image_name", "caption_text"]
    df["caption_text"] = df["caption_text"].astype(str).str.lower().str.strip()

    # -- Dataset & Vocab
    dataset = CaptionDataset(IMAGES_DIR, df, vocab=None, transform=img_transform, freq_threshold=FREQ_THRESHOLD)
    vocab = dataset.vocab
    vocab_size = len(vocab)
    print(f"Dataset size: {len(dataset)}  Vocab size: {vocab_size}")

    # -- Split
    total = len(dataset)
    if total == 0:
        raise RuntimeError("No dataset items. Check images and captions.")
    tr = int(0.8 * total)
    ts = total - tr
    train_set, test_set = random_split(dataset, [tr, ts])
    pad_idx = vocab.stoi["<PAD>"]

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=PadCollate(pad_idx), num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=PadCollate(pad_idx), num_workers=NUM_WORKERS)

    # -- Model, optimizer, loss
    model = ImageCaptionNet(EMBED_DIM, vocab_size, nhead=NUM_HEADS, num_layers=NUM_DECODER_LAYERS).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-5)

    # -- Train
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch}")
        for i, (images, captions) in pbar:
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images, captions, pad_idx=pad_idx)   # (B, T, V)
            logits = outputs[:, :-1, :].contiguous()            # predict tokens 1..T given 0..T-1
            targets = captions[:, 1:].contiguous()
            loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % PRINT_EVERY == 0:
                pbar.set_postfix(avg_loss = running_loss / (i+1))
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch} training loss: {avg_loss:.4f}")

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, captions in tqdm(test_loader, desc="Validation"):
                images = images.to(DEVICE)
                captions = captions.to(DEVICE)
                outputs = model(images, captions, pad_idx=pad_idx)
                logits = outputs[:, :-1, :].contiguous()
                targets = captions[:, 1:].contiguous()
                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                val_loss += loss.item()
        val_loss /= len(test_loader)
        print(f"Epoch {epoch} validation loss: {val_loss:.4f}")

    # -- Evaluation: BLEU on test set (greedy)
    model.eval()
    bleu_scores = []
    with torch.no_grad():
        for images, captions in tqdm(test_loader, desc="Evaluating BLEU"):
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)
            for i in range(images.size(0)):
                img = images[i].unsqueeze(0)
                pred_tokens = model.generate_caption(img, vocab, max_length=MAX_GEN_LEN, device=DEVICE)
                # reference from captions tensor -> decode ids
                ref_ids = captions[i].cpu().numpy().tolist()
                ref_words = [vocab.itos.get(id_, "<UNK>") for id_ in ref_ids
                             if id_ not in (vocab.stoi["<PAD>"], vocab.stoi["<SOS>"], vocab.stoi["<EOS>"])]
                bleu_val = compute_bleu_score([ref_words], pred_tokens)
                bleu_scores.append(bleu_val)
    avg_bleu = float(np.mean(bleu_scores)) if bleu_scores else 0.0
    print(f"\nAverage BLEU score (greedy): {avg_bleu:.4f}")

    # -- Quick sample inference from test set
    if len(test_set) > 0:
        sample_img, sample_cap = test_set[0]
        sample_img_tensor = sample_img.unsqueeze(0).to(DEVICE)
        pred = model.generate_caption(sample_img_tensor, vocab, max_length=MAX_GEN_LEN, device=DEVICE)
        print("\nSample generated caption:", " ".join(pred))
        ref_ids = sample_cap.cpu().numpy().tolist()
        ref_words = [vocab.itos.get(id_, "<UNK>") for id_ in ref_ids if id_ not in (vocab.stoi["<PAD>"], vocab.stoi["<SOS>"], vocab.stoi["<EOS>"])]
        print("Reference:", " ".join(ref_words))

if __name__ == "__main__":
    main()
