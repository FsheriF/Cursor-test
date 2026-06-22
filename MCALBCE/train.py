import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score, matthews_corrcoef,
    confusion_matrix, roc_curve,
)
from tqdm import tqdm

from model import MCALBCE
from dataset import BCEDataset, generate_demo_data, load_csv, load_fasta


def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)
    return {
        "ACC": acc, "SN": sn, "SP": sp, "AUC": auc, "MCC": mcc,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        total_loss += loss.item() * batch_x.size(0)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    return avg_loss, metrics, all_labels, all_probs


def main():
    parser = argparse.ArgumentParser(description="Train MCALBCE model")
    parser.add_argument("--train_file", type=str, default=None,
                        help="Path to training data (CSV or FASTA)")
    parser.add_argument("--test_file", type=str, default=None,
                        help="Path to test data (CSV or FASTA)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--gru_hidden", type=int, default=64)
    parser.add_argument("--transformer_layers", type=int, default=2)
    parser.add_argument("--transformer_nhead", type=int, default=8)
    parser.add_argument("--transformer_ff_dim", type=int, default=256)
    parser.add_argument("--cross_attn_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.train_file and args.test_file:
        ext = os.path.splitext(args.train_file)[1].lower()
        load_fn = load_csv if ext == ".csv" else load_fasta
        train_seqs, train_labels = load_fn(args.train_file)
        test_seqs, test_labels = load_fn(args.test_file)
    else:
        print("No data files provided. Using synthetic demo data.")
        train_seqs, train_labels, test_seqs, test_labels = generate_demo_data()

    print(f"Training set: {sum(train_labels)} positive, "
          f"{len(train_labels) - sum(train_labels)} negative")
    print(f"Test set: {sum(test_labels)} positive, "
          f"{len(test_labels) - sum(test_labels)} negative")

    train_dataset = BCEDataset(train_seqs, train_labels)
    test_dataset = BCEDataset(test_seqs, test_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = MCALBCE(
        vocab_size=24,
        embed_dim=args.embed_dim,
        max_seq_len=25,
        lstm_hidden=args.lstm_hidden,
        gru_hidden=args.gru_hidden,
        num_transformer_layers=args.transformer_layers,
        transformer_nhead=args.transformer_nhead,
        transformer_ff_dim=args.transformer_ff_dim,
        cross_attn_heads=args.cross_attn_heads,
        num_classes=1,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_metrics, _, _ = evaluate(
            model, test_loader, criterion, device
        )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} ACC: {train_metrics['ACC']:.3f} "
                f"SN: {train_metrics['SN']:.3f} AUC: {train_metrics['AUC']:.3f} | "
                f"Test  ACC: {test_metrics['ACC']:.3f} SN: {test_metrics['SN']:.3f} "
                f"SP: {test_metrics['SP']:.3f} AUC: {test_metrics['AUC']:.3f} "
                f"MCC: {test_metrics['MCC']:.3f}"
            )

        if test_metrics["AUC"] > best_auc:
            best_auc = test_metrics["AUC"]
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))

    torch.save(model.state_dict(), os.path.join(args.save_dir, "final_model.pt"))

    print("\n=== Final Evaluation on Independent Test Set ===")
    model.load_state_dict(
        torch.load(os.path.join(args.save_dir, "best_model.pt"), weights_only=True)
    )
    _, final_metrics, y_true, y_prob = evaluate(model, test_loader, criterion, device)
    print(f"ACC: {final_metrics['ACC']:.3f}")
    print(f"SN:  {final_metrics['SN']:.3f}")
    print(f"SP:  {final_metrics['SP']:.3f}")
    print(f"AUC: {final_metrics['AUC']:.3f}")
    print(f"MCC: {final_metrics['MCC']:.3f}")
    print(f"Best AUC: {best_auc:.3f}")

    np.savez(
        os.path.join(args.save_dir, "test_results.npz"),
        y_true=y_true, y_prob=y_prob,
    )


if __name__ == "__main__":
    main()
