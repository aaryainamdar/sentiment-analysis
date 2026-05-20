"""
utils/eda.py
Exploratory Data Analysis and visualizations for the sentiment dataset.
Generates:
  - Class distribution pie + bar chart
  - Review length distribution
  - Word cloud per sentiment class
  - Top N-grams per class
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

COLORS = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}


def plot_class_distribution(df: pd.DataFrame):
    """Pie + bar chart of sentiment label counts."""
    counts = df["sentiment_label"].value_counts()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Pie chart
    ax1.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        colors=[COLORS.get(l, "grey") for l in counts.index],
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax1.set_title("Sentiment Class Distribution (Pie)")

    # Bar chart
    bars = ax2.bar(counts.index, counts.values,
                   color=[COLORS.get(l, "grey") for l in counts.index],
                   edgecolor="black")
    for bar, val in zip(bars, counts.values):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 100,
                 str(val), ha="center", fontsize=10)
    ax2.set_title("Sentiment Class Distribution (Count)")
    ax2.set_ylabel("Number of Reviews")

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Class distribution chart saved → {path}")


def plot_review_length(df: pd.DataFrame, text_col: str = "text"):
    """Histogram of review lengths (word count) by sentiment."""
    df = df.copy()
    df["word_count"] = df[text_col].fillna("").apply(lambda x: len(x.split()))

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color in COLORS.items():
        subset = df[df["sentiment_label"] == label]["word_count"]
        ax.hist(subset, bins=50, alpha=0.6, color=color, label=label,
                range=(0, 300))
    ax.set_title("Review Length Distribution by Sentiment")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "review_length_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Review length chart saved → {path}")

    stats = df.groupby("sentiment_label")["word_count"].describe()[["mean", "50%", "max"]]
    print("\nReview length stats by sentiment:")
    print(stats.round(1).to_string())


def plot_wordcloud(df: pd.DataFrame, text_col: str = "clean_text"):
    """Generate one word cloud per sentiment class."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("wordcloud not installed – skipping word clouds.")
        return

    for label, color in COLORS.items():
        subset = df[df["sentiment_label"] == label][text_col].dropna()
        text = " ".join(subset.tolist())
        if not text.strip():
            continue
        wc = WordCloud(
            width=800, height=400,
            background_color="white",
            colormap="RdYlGn" if label == "Positive" else ("Reds" if label == "Negative" else "YlOrBr"),
            max_words=150,
        ).generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud – {label} Reviews", fontsize=14)
        path = os.path.join(RESULTS_DIR, f"wordcloud_{label.lower()}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Word cloud saved → {path}")


def plot_top_ngrams(df: pd.DataFrame, text_col: str = "clean_text",
                    n: int = 2, top_k: int = 15):
    """Bar charts of top N-grams per sentiment class."""
    from sklearn.feature_extraction.text import CountVectorizer

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, label in zip(axes, ["Positive", "Neutral", "Negative"]):
        subset = df[df["sentiment_label"] == label][text_col].dropna()
        if len(subset) == 0:
            continue
        vec = CountVectorizer(ngram_range=(n, n), max_features=top_k * 2, min_df=2)
        X = vec.fit_transform(subset)
        freqs = X.sum(axis=0).A1
        terms = vec.get_feature_names_out()
        top = sorted(zip(terms, freqs), key=lambda x: x[1], reverse=True)[:top_k]
        terms_top, freqs_top = zip(*top)
        ax.barh(list(reversed(terms_top)), list(reversed(freqs_top)),
                color=COLORS[label], edgecolor="black")
        ax.set_title(f"Top {n}-grams – {label}")
        ax.set_xlabel("Frequency")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"top_{n}grams.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Top {n}-gram chart saved → {path}")


def run_eda(df: pd.DataFrame, text_col: str = "text", clean_col: str = "clean_text"):
    """Run full EDA pipeline."""
    print("\n" + "="*50)
    print("Exploratory Data Analysis")
    print("="*50)
    plot_class_distribution(df)
    plot_review_length(df, text_col=text_col)
    if clean_col in df.columns:
        plot_wordcloud(df, text_col=clean_col)
        plot_top_ngrams(df, text_col=clean_col, n=1)
        plot_top_ngrams(df, text_col=clean_col, n=2)
    print("EDA complete.\n")