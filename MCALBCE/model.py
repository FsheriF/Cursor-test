import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=25, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention: Q from BiLSTM, K from BiGRU, V from Transformer."""

    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) * V
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )
        output = self.W_O(attn_output)
        return output, attn_weights


class MCALBCE(nn.Module):
    """
    MCALBCE: Multi-head Cross-Attention for Linear B-Cell Epitope prediction.

    Architecture:
        1. Embedding: amino acid index -> 128-dim vector
        2. BiLSTM branch -> Query
        3. BiGRU branch -> Key
        4. Transformer Encoder branch -> Value
        5. Multi-head Cross-Attention (8 heads) fuses Q, K, V
        6. Fully connected classifier -> binary output
    """

    def __init__(
        self,
        vocab_size=24,       # 0=padding, 1-23=amino acids
        embed_dim=128,
        max_seq_len=25,
        lstm_hidden=64,      # BiLSTM: 64*2=128 to match embed_dim
        gru_hidden=64,       # BiGRU: 64*2=128 to match embed_dim
        num_transformer_layers=2,
        transformer_nhead=8,
        transformer_ff_dim=256,
        cross_attn_heads=8,
        num_classes=1,
        dropout=0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        # 1. Word Embedding Module
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 2. BiLSTM Module (output -> Query)
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.bilstm_norm = nn.LayerNorm(lstm_hidden * 2)

        # 3. BiGRU Module (output -> Key)
        self.bigru = nn.GRU(
            input_size=embed_dim,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.bigru_norm = nn.LayerNorm(gru_hidden * 2)

        # 4. Transformer Encoder Module (output -> Value)
        self.pos_encoder = PositionalEncoding(embed_dim, max_seq_len, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_nhead,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )
        self.transformer_norm = nn.LayerNorm(embed_dim)

        # 5. Multi-Head Cross-Attention Module
        self.cross_attention = MultiHeadCrossAttention(
            d_model=embed_dim, num_heads=cross_attn_heads, dropout=dropout
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)

        # 6. Classification Module
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        # x: (batch, seq_len) integer-encoded amino acids

        # Embedding: (batch, seq_len) -> (batch, seq_len, 128)
        x_emb = self.embedding(x)

        # BiLSTM branch -> Query
        lstm_out, _ = self.bilstm(x_emb)
        lstm_out = self.bilstm_norm(lstm_out)

        # BiGRU branch -> Key
        gru_out, _ = self.bigru(x_emb)
        gru_out = self.bigru_norm(gru_out)

        # Transformer Encoder branch -> Value
        trans_in = self.pos_encoder(x_emb)
        trans_out = self.transformer_encoder(trans_in)
        trans_out = self.transformer_norm(trans_out)

        # Multi-Head Cross-Attention: Q=BiLSTM, K=BiGRU, V=Transformer
        cross_out, attn_weights = self.cross_attention(
            query=lstm_out, key=gru_out, value=trans_out
        )
        cross_out = self.cross_attn_norm(cross_out)

        # Global average pooling over sequence dimension
        pooled = cross_out.mean(dim=1)

        # Classification
        logits = self.classifier(pooled)
        return logits.squeeze(-1)

    def get_features(self, x):
        """Extract features before the classifier (for t-SNE visualization)."""
        x_emb = self.embedding(x)
        lstm_out, _ = self.bilstm(x_emb)
        lstm_out = self.bilstm_norm(lstm_out)
        gru_out, _ = self.bigru(x_emb)
        gru_out = self.bigru_norm(gru_out)
        trans_in = self.pos_encoder(x_emb)
        trans_out = self.transformer_encoder(trans_in)
        trans_out = self.transformer_norm(trans_out)
        cross_out, _ = self.cross_attention(
            query=lstm_out, key=gru_out, value=trans_out
        )
        cross_out = self.cross_attn_norm(cross_out)
        pooled = cross_out.mean(dim=1)
        return pooled
