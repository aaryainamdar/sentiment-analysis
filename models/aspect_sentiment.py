
"""
models/aspect_sentiment.py
Aspect-Based Sentiment Analysis (ABSA)

Extracts opinions about specific aspects of a product:
  - Quality, Price, Delivery, Packaging, Taste (food-specific), Service

Uses VADER for per-aspect sentiment scoring and keyword matching.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

VADER = SentimentIntensityAnalyzer()

# Aspect keyword dictionary
ASPECT_KEYWORDS = {
    "Quality": [
        "quality", "good", "bad", "excellent", "poor", "durable",
        "sturdy", "cheap", "broken", "defective", "premium", "solid",
        "flimsy", "well made", "well-made",
    ],
    "Price": [
        "price", "cost", "expensive", "cheap", "affordable", "value",
        "worth", "overpriced", "deal", "money", "pricey", "budget",
    ],
    "Delivery": [
        "delivery", "shipping", "arrived", "fast", "slow", "late",
        "on time", "package", "transit", "delayed", "quickly",
    ],
    "Packaging": [
        "packaging", "box", "packed", "wrapped", "container",
        "bottle", "seal", "damaged", "intact",
    ],
    "Taste": [
        "taste", "flavor", "delicious", "yummy", "disgusting",
        "fresh", "stale", "sweet", "sour", "bland", "spicy",
        "smell", "aroma",
    ],
    "Service": [
        "service", "support", "customer", "helpful", "rude",
        "response", "refund", "return", "seller", "amazon",
    ],
}


def extract_aspect_sentences(text: str, aspect: str) -> list:
    """Split text into sentences and return those mentioning an aspect."""
    keywords = ASPECT_KEYWORDS.get(aspect, [])
    sentences = re.split(r"[.!?]+", text.lower())
    matched = []
    for sent in sentences:
        if any(kw in sent for kw in keywords):
            matched.append(sent.strip())
    return matched


def score_aspect(text: str, aspect: str) -> dict:
    """
    Return VADER compound score for the subset of text mentioning the aspect.
    Returns None if aspect is not mentioned.
    """
    sentences = extract_aspect_sentences(text, aspect)
    if not sentences:
        return {"aspect": aspect, "mentioned": False, "score": None, "label": None}

    combined = " ".join(sentences)
    score = VADER.polarity_scores(combined)["compound"]

    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {"aspect": aspect, "mentioned": True, "score": round(score, 4), "label": label}


def analyze_aspects(text: str) -> pd.DataFrame:
    """
    Analyze all aspects for a single review.
    Returns a DataFrame with one row per aspect.
    """
    results = [score_aspect(text, asp) for asp in ASPECT_KEYWORDS]
    return pd.DataFrame(results)


def batch_aspect_analysis(df: pd.DataFrame, text_col: str = "text",
                           sample_n: int = 2000) -> pd.DataFrame:
    """
    Run aspect analysis over a sample of reviews.
    Returns aggregated sentiment per aspect.
    """
    sample = df.sample(min(sample_n, len(df)), random_state=42)
    records = []
    for _, row in sample.iterrows():
        for asp in ASPECT_KEYWORDS:
            result = score_aspect(row[text_col], asp)
            if result["mentioned"]:
                records.append(result)

    aspect_df = pd.DataFrame(records)
    return aspect_df


def plot_aspect_sentiment(aspect_df: pd.DataFrame):
    """Bar chart: % Positive / Neutral / Negative per aspect."""
    pivot = (
        aspect_df.groupby(["aspect", "label"])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure all label columns exist
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    colors = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
    ax = pivot_pct[["Positive", "Neutral", "Negative"]].plot(
        kind="bar", figsize=(10, 6),
        color=[colors["Positive"], colors["Neutral"], colors["Negative"]],
        edgecolor="black",
    )
    ax.set_title("Aspect-Based Sentiment Analysis", fontsize=14)
    ax.set_xlabel("Aspect")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "aspect_sentiment.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Aspect sentiment chart saved → {path}")


def plot_aspect_avg_score(aspect_df: pd.DataFrame):
    """Horizontal bar chart of average compound score per aspect."""
    avg = aspect_df.groupby("aspect")["score"].mean().sort_values()
    colors = ["#F44336" if v < 0 else "#4CAF50" for v in avg.values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(avg.index, avg.values, color=colors, edgecolor="black")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Average Sentiment Score per Aspect")
    ax.set_xlabel("VADER Compound Score")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "aspect_avg_score.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Aspect avg score chart saved → {path}")


def run_aspect_analysis(df: pd.DataFrame, text_col: str = "text"):
    """Full pipeline: batch analysis + plots."""
    print("\nRunning Aspect-Based Sentiment Analysis...")
    aspect_df = batch_aspect_analysis(df, text_col=text_col)
    plot_aspect_sentiment(aspect_df)
    plot_aspect_avg_score(aspect_df)

    summary = aspect_df.groupby("aspect")["label"].value_counts(normalize=True).mul(100).round(1)
    print("\nAspect Sentiment Summary (%):")
    print(summary.to_string())
    return aspect_df


if __name__ == "__main__":
    review = (
        "The taste is amazing and the flavor is so fresh! "
        "However the packaging was damaged when it arrived. "
        "Delivery was surprisingly fast. "
        "A bit expensive for the quality but overall good value."
    )
    print("Sample review aspect analysis:")
    print(analyze_aspects(review).to_string(index=False))
