import numpy as np
import torch
from torch.utils.data import Dataset

AMINO_ACIDS = {
    'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
    'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
    'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
    'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
    'X': 21, 'U': 22, 'O': 23,
}

MAX_SEQ_LEN = 25
NUM_AMINO_ACIDS = 23


def encode_sequence(seq, max_len=MAX_SEQ_LEN):
    encoded = []
    for aa in seq.upper():
        if aa in AMINO_ACIDS:
            encoded.append(AMINO_ACIDS[aa])
        else:
            encoded.append(0)
    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return encoded


class BCEDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        encoded = encode_sequence(seq)
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float),
        )


def load_fasta(filepath):
    sequences = []
    labels = []
    current_seq = ""
    current_label = None
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq and current_label is not None:
                    sequences.append(current_seq)
                    labels.append(current_label)
                current_seq = ""
                header = line.lower()
                if "positive" in header or "linear" in header or "|1" in header:
                    current_label = 1
                else:
                    current_label = 0
            else:
                current_seq += line
    if current_seq and current_label is not None:
        sequences.append(current_seq)
        labels.append(current_label)
    return sequences, labels


def load_csv(filepath):
    import pandas as pd
    df = pd.read_csv(filepath)
    seq_col = None
    label_col = None
    for col in df.columns:
        cl = col.lower()
        if "seq" in cl or "peptide" in cl or "epitope" in cl:
            seq_col = col
        if "label" in cl or "class" in cl or "target" in cl:
            label_col = col
    if seq_col is None:
        seq_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[-1]
    sequences = df[seq_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return sequences, labels


def generate_demo_data():
    """Generate synthetic demo data for testing the model pipeline."""
    np.random.seed(42)
    aa_list = list(AMINO_ACIDS.keys())[:20]

    train_seqs, train_labels = [], []
    test_seqs, test_labels = [], []

    for _ in range(4440):
        length = np.random.randint(5, 26)
        weights = np.ones(20)
        weights[2] += 1.5   # D (Asp) - enriched in epitopes
        weights[11] += 1.2  # N (Asn) - enriched in epitopes
        weights[9] -= 0.5   # L (Leu) - depleted in epitopes
        weights = weights / weights.sum()
        seq = "".join(np.random.choice(aa_list, size=length, p=weights))
        train_seqs.append(seq)
        train_labels.append(1)

    for _ in range(5485):
        length = np.random.randint(5, 26)
        weights = np.ones(20)
        weights[9] += 1.5   # L (Leu) - enriched in non-epitopes
        weights[10] += 1.0  # M (Met) - enriched in non-epitopes
        weights[2] -= 0.5   # D (Asp) - depleted in non-epitopes
        weights = weights / weights.sum()
        seq = "".join(np.random.choice(aa_list, size=length, p=weights))
        train_seqs.append(seq)
        train_labels.append(0)

    for _ in range(1110):
        length = np.random.randint(5, 26)
        weights = np.ones(20)
        weights[2] += 1.5
        weights[11] += 1.2
        weights[9] -= 0.5
        weights = weights / weights.sum()
        seq = "".join(np.random.choice(aa_list, size=length, p=weights))
        test_seqs.append(seq)
        test_labels.append(1)

    for _ in range(1408):
        length = np.random.randint(5, 26)
        weights = np.ones(20)
        weights[9] += 1.5
        weights[10] += 1.0
        weights[2] -= 0.5
        weights = weights / weights.sum()
        seq = "".join(np.random.choice(aa_list, size=length, p=weights))
        test_seqs.append(seq)
        test_labels.append(0)

    return train_seqs, train_labels, test_seqs, test_labels
