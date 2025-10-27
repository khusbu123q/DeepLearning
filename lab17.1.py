import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from Bio import SeqIO

# Amino Acid Vocabulary with padding
AMINO_ACID_TO_IDX = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X': 21,  # Unknown
    '<PAD>': 0
}

FIXED_MAX_LEN = 512  # maximum sequence length


def parse_fasta_sequences(fasta_path):
    seqs = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        pid = record.id.split('|')[1] if '|' in record.id else record.id
        seqs[pid] = str(record.seq)
    return seqs


def parse_go_annotations(annotation_path):
    go_map = defaultdict(set)
    df = pd.read_csv(annotation_path, sep='\t', header=None, names=['protein_id', 'go_term', 'ontology'])
    for _, row in df.iterrows():
        go_map[row['protein_id']].add(row['go_term'])
    return go_map


def generate_go_term_indices(go_annotations):
    unique_terms = set(term for terms in go_annotations.values() for term in terms)
    idx_map = {term: idx for idx, term in enumerate(sorted(unique_terms))}
    return idx_map


def encode_protein_sequence(sequence, max_length=FIXED_MAX_LEN):
    encoded = [AMINO_ACID_TO_IDX.get(residue, AMINO_ACID_TO_IDX['X']) for residue in sequence[:max_length]]
    padding = [AMINO_ACID_TO_IDX['<PAD>']] * (max_length - len(encoded))
    return encoded + padding


def encode_go_labels(protein_id, annotation_map, go_term_to_idx):
    labels = np.zeros(len(go_term_to_idx), dtype=np.float32)
    for go_term in annotation_map.get(protein_id, []):
        if go_term in go_term_to_idx:
            labels[go_term_to_idx[go_term]] = 1.0
    return labels


class ProteinFunctionDataset(Dataset):
    def __init__(self, seqs, annotations, term_indices):
        self.protein_ids = list(seqs.keys())
        self.sequences = seqs
        self.annotations = annotations
        self.term_to_idx = term_indices

    def __len__(self):
        return len(self.protein_ids)

    def __getitem__(self, idx):
        pid = self.protein_ids[idx]
        seq_encoded = encode_protein_sequence(self.sequences[pid])
        label_vec = encode_go_labels(pid, self.annotations, self.term_to_idx)
        return {
            'seq_tensor': torch.LongTensor(seq_encoded),
            'label_tensor': torch.FloatTensor(label_vec),
            'protein_id': pid
        }


class ProteinBiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, layers=2, dropout_p=0.3):
        super(ProteinBiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.bilstm = nn.LSTM(emb_dim, hidden_dim, num_layers=layers,
                              batch_first=True, bidirectional=True,
                              dropout=dropout_p if layers > 1 else 0)
        self.dropout_layer = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        emb = self.dropout_layer(self.embedding(x))
        lstm_out, _ = self.bilstm(emb)
        pooled_out, _ = torch.max(lstm_out, dim=1)
        dropped = self.dropout_layer(pooled_out)
        out = self.activation(self.fc1(dropped))
        out = self.dropout_layer(out)
        out = self.fc2(out)
        return torch.sigmoid(out)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.
    for batch in loader:
        inputs = batch['seq_tensor'].to(device)
        targets = batch['label_tensor'].to(device)

        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for batch in loader:
            inputs = batch['seq_tensor'].to(device)
            targets = batch['label_tensor'].to(device)
            preds = model(inputs)
            loss = criterion(preds, targets)
            total_loss += loss.item()
    return total_loss / len(loader)


def predict_go_terms_for_protein(model, sequence, device, term_indices,
                                threshold=0.01, max_seq_len=FIXED_MAX_LEN):
    model.eval()
    encoded_seq = encode_protein_sequence(sequence, max_seq_len)
    input_tensor = torch.LongTensor([encoded_seq]).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)[0].cpu().numpy()

    idx_to_term = {idx: term for term, idx in term_indices.items()}
    preds = [(idx_to_term[i], prob) for i, prob in enumerate(outputs) if prob >= threshold]
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds


def generate_predictions_and_save(model, test_fasta, term_indices, out_path,
                                  device, threshold=0.01, max_preds=1500):
    test_seqs = parse_fasta_sequences(test_fasta)

    with open(out_path, 'w') as outf:
        for pid, seq in test_seqs.items():
            predictions = predict_go_terms_for_protein(model, seq, device,
                                                       term_indices, threshold=threshold)
            for term, score in predictions[:max_preds]:
                outf.write(f"{pid}\t{term}\t{score:.4f}\n")
    print(f"Predictions saved to {out_path}")


def main():
    base_dir = '/home/ibab/Downloads/'
    train_seq_path = os.path.join(base_dir, 'cafa-6-protein-function-prediction/Train/train_sequences.fasta')
    train_go_terms_path = os.path.join(base_dir, 'cafa-6-protein-function-prediction/Train/train_terms.tsv')
    test_seq_path = os.path.join(base_dir, 'cafa-6-protein-function-prediction/Test/testsuperset.fasta')
    predictions_out_path = os.path.join(base_dir, 'bilstm_predictions.tsv')

    print("Loading training sequences and GO annotations...")
    train_seqs = parse_fasta_sequences(train_seq_path)
    train_annotations = parse_go_annotations(train_go_terms_path)
    go_term_idx_map = generate_go_term_indices(train_annotations)
    print(f"Total training proteins: {len(train_seqs)}, GO terms: {len(go_term_idx_map)}")

    dataset = ProteinFunctionDataset(train_seqs, train_annotations, go_term_idx_map)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = ProteinBiLSTM(
        vocab_size=len(AMINO_ACID_TO_IDX),
        emb_dim=128,
        hidden_dim=256,
        output_dim=len(go_term_idx_map),
        layers=2,
        dropout_p=0.3
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    epochs = 20
    best_val_loss = float('inf')
    best_model_path = 'best_bilstm_protein_model.pth'

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_model(model, val_loader, criterion, device)
        lr_scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch + 1}")

    print("Training complete.")

    # Load best model and run predictions on test dataset
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_predictions_and_save(model, test_seq_path, go_term_idx_map,
                                  predictions_out_path, device)

if __name__ == "__main__":
    main()
