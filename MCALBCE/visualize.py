"""
Visualization utilities for MCALBCE model analysis:
- ROC curve
- t-SNE visualization (before/after feature extraction)
- In-silico mutagenesis (ISM) heatmap
- Amino acid frequency statistics
- SHAP-like feature contribution analysis
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

from model import MCALBCE
from dataset import BCEDataset, generate_demo_data, encode_sequence, AMINO_ACIDS, MAX_SEQ_LEN


AMINO_ACID_LIST = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
]


def plot_roc_curve(y_true, y_prob, save_path="figures/roc_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"MCALBCE (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve - MCALBCE", fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to {save_path}")


def plot_tsne(model, dataloader, device, save_path="figures/tsne.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    features_before, features_after, labels = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            emb = model.embedding(batch_x)
            emb_pooled = emb.mean(dim=1).cpu().numpy()
            features_before.append(emb_pooled)

            feat = model.get_features(batch_x).cpu().numpy()
            features_after.append(feat)
            labels.append(batch_y.numpy())

    features_before = np.concatenate(features_before, axis=0)
    features_after = np.concatenate(features_after, axis=0)
    labels = np.concatenate(labels, axis=0)

    n_samples = min(2000, len(labels))
    idx = np.random.choice(len(labels), n_samples, replace=False)
    features_before = features_before[idx]
    features_after = features_after[idx]
    labels_sub = labels[idx]

    tsne_before = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(features_before)
    tsne_after = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(features_after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#3498db" if l == 1 else "#e74c3c" for l in labels_sub]

    ax1.scatter(tsne_before[:, 0], tsne_before[:, 1], c=colors, alpha=0.5, s=10)
    ax1.set_title("Before Feature Extraction", fontsize=13)
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")

    ax2.scatter(tsne_after[:, 0], tsne_after[:, 1], c=colors, alpha=0.5, s=10)
    ax2.set_title("After Feature Extraction", fontsize=13)
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3498db', markersize=8, label='Linear BCE'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c', markersize=8, label='Non-linear BCE'),
    ]
    ax2.legend(handles=legend_elements, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"t-SNE visualization saved to {save_path}")


def in_silico_mutagenesis(model, sequence, device, save_path="figures/ism_heatmap.png"):
    """Perform in-silico mutagenesis on a single sequence and plot a heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    original_encoded = encode_sequence(sequence)
    original_tensor = torch.tensor([original_encoded], dtype=torch.long).to(device)
    with torch.no_grad():
        original_prob = torch.sigmoid(model(original_tensor)).item()

    seq_len = len(sequence)
    score_matrix = np.zeros((20, seq_len))

    for pos in range(seq_len):
        for aa_idx, aa in enumerate(AMINO_ACID_LIST):
            mutant_seq = list(sequence)
            mutant_seq[pos] = aa
            mutant_encoded = encode_sequence("".join(mutant_seq))
            mutant_tensor = torch.tensor([mutant_encoded], dtype=torch.long).to(device)
            with torch.no_grad():
                mutant_prob = torch.sigmoid(model(mutant_tensor)).item()
            score_matrix[aa_idx, pos] = mutant_prob

    plt.figure(figsize=(max(12, seq_len * 0.6), 8))
    sns.heatmap(
        score_matrix,
        xticklabels=[f"{sequence[i]}{i+1}" for i in range(seq_len)],
        yticklabels=AMINO_ACID_LIST,
        cmap="RdYlBu_r",
        center=0.5,
        vmin=0, vmax=1,
        linewidths=0.5,
    )
    plt.xlabel("Position", fontsize=12)
    plt.ylabel("Substituted Amino Acid", fontsize=12)
    plt.title(f"In-Silico Mutagenesis (Original prob: {original_prob:.3f})", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ISM heatmap saved to {save_path}")


def plot_aa_frequency(train_seqs, train_labels, save_path="figures/aa_frequency.png"):
    """Plot amino acid frequency for positive and negative samples at each position."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pos_seqs = [s for s, l in zip(train_seqs, train_labels) if l == 1]
    neg_seqs = [s for s, l in zip(train_seqs, train_labels) if l == 0]

    def compute_freq(seqs):
        freq = np.zeros(20)
        total = 0
        for seq in seqs:
            for aa in seq:
                if aa in AMINO_ACID_LIST:
                    freq[AMINO_ACID_LIST.index(aa)] += 1
                    total += 1
        return freq / total if total > 0 else freq

    pos_freq = compute_freq(pos_seqs)
    neg_freq = compute_freq(neg_seqs)

    x = np.arange(20)
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width / 2, pos_freq, width, label="Linear BCE", color="#3498db", alpha=0.8)
    bars2 = ax.bar(x + width / 2, neg_freq, width, label="Non-linear BCE", color="#e74c3c", alpha=0.8)
    ax.set_xlabel("Amino Acid", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Amino Acid Frequency Distribution", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(AMINO_ACID_LIST)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Amino acid frequency plot saved to {save_path}")


def shap_like_analysis(model, sequence, device, save_path="figures/shap_values.png"):
    """Approximate feature contribution by leave-one-out perturbation."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()

    original_encoded = encode_sequence(sequence)
    original_tensor = torch.tensor([original_encoded], dtype=torch.long).to(device)
    with torch.no_grad():
        original_prob = torch.sigmoid(model(original_tensor)).item()

    contributions = []
    positions = []
    for pos in range(len(sequence)):
        perturbed = list(original_encoded)
        perturbed[pos] = 0  # mask position
        perturbed_tensor = torch.tensor([perturbed], dtype=torch.long).to(device)
        with torch.no_grad():
            perturbed_prob = torch.sigmoid(model(perturbed_tensor)).item()
        contribution = original_prob - perturbed_prob
        contributions.append(contribution)
        positions.append(f"{sequence[pos]}{pos+1}")

    colors = ["#e74c3c" if c < 0 else "#3498db" for c in contributions]
    plt.figure(figsize=(max(10, len(sequence) * 0.5), 5))
    plt.bar(range(len(contributions)), contributions, color=colors)
    plt.xticks(range(len(positions)), positions, rotation=45, ha="right")
    plt.xlabel("Amino Acid Position", fontsize=12)
    plt.ylabel("Contribution (SHAP-like)", fontsize=12)
    plt.title("Feature Contribution Analysis", fontsize=14)
    plt.axhline(y=0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"SHAP-like analysis saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize MCALBCE results")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--results_path", type=str, default="checkpoints/test_results.npz")
    parser.add_argument("--save_dir", type=str, default="figures")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_seqs, train_labels, test_seqs, test_labels = generate_demo_data()

    if os.path.exists(args.results_path):
        data = np.load(args.results_path)
        y_true, y_prob = data["y_true"], data["y_prob"]
        plot_roc_curve(y_true, y_prob, os.path.join(args.save_dir, "roc_curve.png"))

    model = MCALBCE().to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
        print("Loaded model from checkpoint.")
    else:
        print("No checkpoint found, using randomly initialized model for visualization demo.")

    test_dataset = BCEDataset(test_seqs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    plot_tsne(model, test_loader, device, os.path.join(args.save_dir, "tsne.png"))
    plot_aa_frequency(train_seqs, train_labels, os.path.join(args.save_dir, "aa_frequency.png"))

    example_seq = test_seqs[0]
    print(f"Example sequence for ISM: {example_seq}")
    in_silico_mutagenesis(model, example_seq, device, os.path.join(args.save_dir, "ism_heatmap.png"))
    shap_like_analysis(model, example_seq, device, os.path.join(args.save_dir, "shap_values.png"))


if __name__ == "__main__":
    main()
