import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from Bio import SeqIO
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

# Amino acid vocabulary (integer encoding)
AMINO_ACID_INDEX = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X': 21,  # Unknown amino acid
    '<PAD>': 0  # Padding token
}

MAX_PROTEIN_LENGTH = 512  # maximum sequence length


def parse_fasta(fasta_path):
    protein_seqs = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        pid = record.id.split('|')[1] if '|' in record.id else record.id
        protein_seqs[pid] = str(record.seq)
    return protein_seqs


def parse_go_labels(tsv_path):
    protein_to_terms = defaultdict(set)
    df = pd.read_csv(tsv_path, sep='\t', header=None, names=['protein_id', 'go_term', 'ontology'])
    for _, entry in df.iterrows():
        protein_to_terms[entry['protein_id']].add(entry['go_term'])
    return protein_to_terms


def create_go_term_dict(go_annotation_map):
    all_go_terms = {term for terms in go_annotation_map.values() for term in terms}
    term_index = {term: idx for idx, term in enumerate(sorted(all_go_terms))}
    return term_index


def sequence_to_indices(seq, max_len=MAX_PROTEIN_LENGTH):
    seq = seq[:max_len]
    encoded = [AMINO_ACID_INDEX.get(residue, AMINO_ACID_INDEX['X']) for residue in seq]
    padded = encoded + [AMINO_ACID_INDEX['<PAD>']] * (max_len - len(encoded))
    return padded


def encode_protein_labels(protein_id, annotation_map, go_term_dict):
    labels = np.zeros(len(go_term_dict), dtype=np.float32)
    if protein_id in annotation_map:
        for go_term in annotation_map[protein_id]:
            if go_term in go_term_dict:
                labels[go_term_dict[go_term]] = 1.0
    return labels


class ProteinFunctionDatasetCustom(Dataset):
    def __init__(self, protein_sequences, protein_annotations, go_index):
        self.protein_ids = list(protein_sequences.keys())
        self.sequences = protein_sequences
        self.annotations = protein_annotations
        self.term_to_idx = go_index

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        seq_indices = sequence_to_indices(self.sequences[pid])
        labels_vec = encode_protein_labels(pid, self.annotations, self.term_to_idx)
        return {
            'seq_tensor': torch.LongTensor(seq_indices),
            'label_tensor': torch.FloatTensor(labels_vec),
            'protein_id': pid
        }


class PositionalEncodingCustom(nn.Module):
    def __init__(self, d_model, max_len=MAX_PROTEIN_LENGTH, dropout_rate=0.1):
        super(PositionalEncodingCustom, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ProteinTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 dim_feedforward, num_classes, max_len=MAX_PROTEIN_LENGTH,
                 dropout=0.3):
        super(ProteinTransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.positional_encoding = PositionalEncodingCustom(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dense1 = nn.Linear(d_model, d_model // 2)
        self.dense2 = nn.Linear(d_model // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, input_seq):
        # input_seq shape: (batch_size, seq_len)
        padding_mask = (input_seq == 0)

        embed = self.embedding(input_seq) * np.sqrt(self.embedding.embedding_dim)
        embed = self.positional_encoding(embed)

        encoded = self.transformer_encoder(embed, src_key_padding_mask=padding_mask)

        mask = (~padding_mask).unsqueeze(-1).float()
        pooled = (encoded * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-10)

        out = self.dropout(pooled)
        out = self.activation(self.dense1(out))
        out = self.dropout(out)
        out = self.dense2(out)
        return torch.sigmoid(out)  # For multi-label classification


def one_epoch_train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        sequences = batch['seq_tensor'].to(device)
        targets = batch['label_tensor'].to(device)

        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def one_epoch_validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch['seq_tensor'].to(device)
            targets = batch['label_tensor'].to(device)
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def predict_go_terms_from_seq(model, sequence, device, term_to_idx, threshold=0.01, max_len=MAX_PROTEIN_LENGTH):
    model.eval()
    encoded_seq = sequence_to_indices(sequence, max_len)
    input_tensor = torch.LongTensor([encoded_seq]).to(device)

    with torch.no_grad():
        output = model(input_tensor)[0].cpu().numpy()

    idx_to_term = {idx: term for term, idx in term_to_idx.items()}
    predictions = [(idx_to_term[i], prob) for i, prob in enumerate(output) if prob >= threshold]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions


def save_predictions(model, fasta_file, go_index, output_path, device, threshold=0.01, max_preds=1500):
    sequences = parse_fasta(fasta_file)
    with open(output_path, 'w') as f_out:
        for prot_id, seq in sequences.items():
            preds = predict_go_terms_from_seq(model, seq, device, go_index, threshold=threshold)
            preds = preds[:max_preds]
            for go_term, prob in preds:
                f_out.write(f"{prot_id}\t{go_term}\t{prob:.4f}\n")
    print(f"Predictions saved to {output_path}")


def main():
    dataset_dir = '/home/ibab/Downloads/'
    train_fasta_path = os.path.join(dataset_dir, 'cafa-6-protein-function-prediction/Train/train_sequences.fasta')
    train_annotations_path = os.path.join(dataset_dir, 'cafa-6-protein-function-prediction/Train/train_terms.tsv')
    test_fasta_path = os.path.join(dataset_dir, 'cafa-6-protein-function-prediction/Test/testsuperset.fasta')
    prediction_save_path = os.path.join(dataset_dir, 'transformer_go_predictions.tsv')

    print("Loading sequences and GO annotations...")
    train_sequences = parse_fasta(train_fasta_path)
    train_go_annotations = parse_go_labels(train_annotations_path)
    go_term_idx_map = create_go_term_dict(train_go_annotations)
    print(f"Loaded {len(train_sequences)} sequences and {len(go_term_idx_map)} GO terms.")

    dataset = ProteinFunctionDatasetCustom(train_sequences, train_go_annotations, go_term_idx_map)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_data, val_data = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = ProteinTransformerModel(
        vocab_size=len(AMINO_ACID_INDEX),
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        num_classes=len(go_term_idx_map),
        max_len=MAX_PROTEIN_LENGTH,
        dropout=0.3
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    epochs = 20
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = one_epoch_train(model, train_loader, criterion, optimizer, device)
        val_loss = one_epoch_validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'protein_transformer_model_best.pth')
            print(f"Saved best model at epoch {epoch+1}")

    print("Training finished.")

    model.load_state_dict(torch.load('protein_transformer_model_best.pth', map_location=device))
    save_predictions(model, test_fasta_path, go_term_idx_map, prediction_save_path, device)


if __name__ == "__main__":
    main()
