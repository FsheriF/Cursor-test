"""
Predict whether input amino acid sequences are linear B-cell epitopes.
"""
import argparse
import torch
import numpy as np

from model import MCALBCE
from dataset import encode_sequence


def predict_sequences(model, sequences, device, threshold=0.5):
    model.eval()
    results = []
    with torch.no_grad():
        for seq in sequences:
            encoded = encode_sequence(seq)
            tensor = torch.tensor([encoded], dtype=torch.long).to(device)
            logit = model(tensor)
            prob = torch.sigmoid(logit).item()
            label = 1 if prob >= threshold else 0
            results.append({
                "sequence": seq,
                "probability": prob,
                "prediction": "Linear BCE" if label == 1 else "Non-linear BCE",
            })
    return results


def main():
    parser = argparse.ArgumentParser(description="Predict linear B-cell epitopes")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--sequences", type=str, nargs="+", required=True,
                        help="Amino acid sequences to predict")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MCALBCE().to(device)
    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )

    results = predict_sequences(model, args.sequences, device, args.threshold)
    print(f"{'Sequence':<30} {'Probability':>12} {'Prediction'}")
    print("-" * 60)
    for r in results:
        print(f"{r['sequence']:<30} {r['probability']:>12.4f} {r['prediction']}")


if __name__ == "__main__":
    main()
