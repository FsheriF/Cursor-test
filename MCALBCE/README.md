# MCALBCE: Multi-head Cross-Attention for Linear B-Cell Epitope Prediction

Reproduction of the MCALBCE model described in:

> 任静怡, 张胜利. 基于多头交叉注意力机制的线性B细胞表位预测. 生物信息学, 2026, 24(2): 127-137.

## Architecture

```
Input Sequence → Embedding (128-dim)
                    ├── BiLSTM → Query (Q)
                    ├── BiGRU  → Key   (K)
                    └── Transformer Encoder → Value (V)
                              ↓
                Multi-Head Cross-Attention (8 heads)
                              ↓
                  Global Average Pooling
                              ↓
                  Fully Connected Classifier
                              ↓
                    Binary Prediction
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Train with demo data
```bash
python train.py
```

### Train with custom data
```bash
python train.py --train_file data/train.csv --test_file data/test.csv
```

CSV files should have a sequence column and a label column (1=linear BCE, 0=non-linear BCE).

### Predict
```bash
python predict.py --model_path checkpoints/best_model.pt --sequences ADGPKRSQD LLLMAVGHI
```

### Visualize
```bash
python visualize.py --model_path checkpoints/best_model.pt
```

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Embedding dim | 128 |
| BiLSTM hidden | 64 (×2 bidirectional) |
| BiGRU hidden | 64 (×2 bidirectional) |
| Transformer layers | 2 |
| Transformer heads | 8 |
| Cross-attention heads | 8 |
| Batch size | 32 |
| Epochs | 100 |
| Learning rate | 0.0001 |
| Optimizer | Adam |
| Max sequence length | 25 |

## Data

The original paper uses IEDB-derived data: 4,440 linear BCEs + 5,485 non-linear BCEs (train), 1,110 + 1,408 (test), filtered by CD-HIT (0.7). Place your data in `data/` or use the built-in synthetic demo data.
