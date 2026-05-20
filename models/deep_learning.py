"""
models/deep_learning.py
Deep Learning sentiment classifier using:
  - Bidirectional LSTM (PyTorch)
  - Pre-trained word embeddings (GloVe optional, random otherwise)

Falls back gracefully if GPU is not available.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from collections import Counter

RESULTS_DIR = "results"
MODELS_DIR = "models/saved"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

LABELS = ["Negative", "Neutral", "Positive"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[DeepLearning] Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 1. Vocabulary builder
# ─────────────────────────────────────────────

class Vocabulary:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, max_vocab: int = 30_000, min_freq: int = 2):
        self.max_vocab = max_vocab
        self.min_freq = min_freq
        self.word2idx = {self.PAD: 0, self.UNK: 1}
        self.idx2word = {0: self.PAD, 1: self.UNK}

    def build(self, texts: list):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        vocab = [w for w, c in counter.most_common(self.max_vocab)
                 if c >= self.min_freq]
        for word in vocab:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        print(f"Vocabulary size: {len(self.word2idx)}")

    def encode(self, text: str, max_len: int = 200) -> list:
        tokens = text.split()[:max_len]
        ids = [self.word2idx.get(t, 1) for t in tokens]  # 1 = UNK
        # Pad or truncate
        ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────
# 2. PyTorch Dataset
# ─────────────────────────────────────────────

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocabulary, max_len: int = 200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(
            self.vocab.encode(self.texts[idx], self.max_len), dtype=torch.long
        )
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ─────────────────────────────────────────────
# 3. BiLSTM Model
# ─────────────────────────────────────────────

class BiLSTMSentiment(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2,
                 num_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (hidden, _) = self.lstm(embedded)
        # Concat last forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(self.dropout(hidden))


# ─────────────────────────────────────────────
# 4. Training loop
# ─────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (out.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            out = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = out.argmax(1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y_batch.cpu().numpy())
    return total_loss / total, correct / total, preds_all, labels_all


# ─────────────────────────────────────────────
# 5. Main training function
# ─────────────────────────────────────────────

def train_lstm(df: pd.DataFrame, text_col: str = "clean_text",
               epochs: int = 10, batch_size: int = 64,
               max_len: int = 200, lr: float = 1e-3):
    """
    Train BiLSTM model on preprocessed reviews.

    Args:
        df        : DataFrame with clean_text and sentiment columns
        text_col  : column to use as features
        epochs    : number of training epochs
        batch_size: mini-batch size
        max_len   : max token length per review
        lr        : learning rate
    Returns:
        Trained model, vocabulary
    """
    texts = df[text_col].fillna("").tolist()
    labels = df["sentiment"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Build vocabulary on training data only
    vocab = Vocabulary(max_vocab=30_000, min_freq=2)
    vocab.build(X_train)

    train_ds = ReviewDataset(X_train, y_train, vocab, max_len)
    test_ds = ReviewDataset(X_test, y_test, vocab, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = BiLSTMSentiment(
        vocab_size=len(vocab),
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        num_classes=3,
        dropout=0.4,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )
    # Class weights to handle imbalance
    class_counts = Counter(y_train)
    weights = torch.tensor(
        [1.0 / class_counts[i] for i in range(3)], dtype=torch.float
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0

    print(f"\nTraining BiLSTM for {epochs} epochs on {DEVICE}...")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, preds, true = eval_epoch(model, test_loader, criterion)
        scheduler.step(val_acc)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(MODELS_DIR, "bilstm_best.pt"))

    # Load best checkpoint and evaluate
    model.load_state_dict(torch.load(
        os.path.join(MODELS_DIR, "bilstm_best.pt"), map_location=DEVICE
    ))
    _, final_acc, final_preds, final_true = eval_epoch(model, test_loader, criterion)

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    print(classification_report(final_true, final_preds, target_names=LABELS))

    # Confusion matrix
    cm = confusion_matrix(final_true, final_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_title("Confusion Matrix – BiLSTM")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "cm_BiLSTM.png"), dpi=150)
    plt.close()

    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["val_loss"], label="Val Loss")
    ax1.set_title("Loss Curves – BiLSTM")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train Acc")
    ax2.plot(history["val_acc"], label="Val Acc")
    ax2.set_title("Accuracy Curves – BiLSTM")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "bilstm_training_curves.png"), dpi=150)
    plt.close()
    print(f"Plots saved to {RESULTS_DIR}/")

    # Save vocab
    import joblib
    joblib.dump(vocab, os.path.join(MODELS_DIR, "lstm_vocab.pkl"))

    return model, vocab


if __name__ == "__main__":
    from utils.data_loader import load_dataset
    from utils.preprocessor import preprocess_dataframe

    df = load_dataset("data/Reviews.csv", sample_size=9000)
    df = preprocess_dataframe(df)
    train_lstm(df, epochs=10)