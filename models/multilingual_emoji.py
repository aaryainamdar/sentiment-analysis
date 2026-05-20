
"""
models/multilingual_emoji.py - Fixed version
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import emoji

try:
    from langdetect import detect, LangDetectException
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    print("Warning: langdetect not installed. Language detection disabled.")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

VADER = SentimentIntensityAnalyzer()

SUPPORTED_LANGUAGES = {
    "en": "English", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "nl": "Dutch",
}

EMOJI_SENTIMENT = {
    "😊": "happy", "😀": "happy", "😍": "love", "❤️": "love",
    "👍": "good", "✅": "good", "🎉": "great", "🌟": "great",
    "😢": "sad", "😭": "cry", "😡": "angry", "👎": "bad",
    "💔": "heartbreak", "😞": "disappointed", "🤮": "disgusting",
    "😲": "surprised", "🤔": "thinking", "😐": "neutral",
}


def detect_language(text: str) -> str:
    if not LANGDETECT_AVAILABLE or not isinstance(text, str) or len(text.strip()) < 10:
        return "unknown"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def score_multilingual(text: str, lang: str = None) -> dict:
    if lang is None:
        lang = detect_language(text)
    if lang == "en":
        score = VADER.polarity_scores(text)["compound"]
    else:
        score = TextBlob(text).sentiment.polarity
    if score >= 0.05:
        label = "Positive"
    elif score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {"language": lang, "score": round(score, 4), "label": label}


def run_multilingual_analysis(df: pd.DataFrame, text_col: str = "text",
                               sample_n: int = 3000) -> pd.DataFrame:
    print("\nRunning Multilingual Sentiment Analysis...")
    sample = df.sample(min(sample_n, len(df)), random_state=42).copy().reset_index(drop=True)

    print("  Detecting languages and scoring sentiment...")
    lang_list, score_list, label_list = [], [], []

    for text in sample[text_col]:
        lang = detect_language(str(text))
        result = score_multilingual(str(text), lang)
        lang_list.append(lang)
        score_list.append(result["score"])
        label_list.append(result["label"])

    sample["language"] = lang_list
    sample["ml_score"] = score_list
    sample["ml_label"] = label_list
    sample["lang_name"] = [
        SUPPORTED_LANGUAGES.get(str(l), f"Other ({l})") for l in lang_list
    ]

    # Plot language distribution
    lang_counts = sample["lang_name"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(9, 4))
    lang_counts.plot(kind="bar", ax=ax, color="#5B9BD5", edgecolor="black")
    ax.set_title("Language Distribution in Reviews")
    ax.set_xlabel("Language")
    ax.set_ylabel("Number of Reviews")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "language_distribution.png"), dpi=150)
    plt.close()

    # Plot sentiment per language (top 5)
    top_langs = lang_counts.head(5).index.tolist()
    subset = sample[sample["lang_name"].isin(top_langs)]
    pivot = subset.groupby(["lang_name", "ml_label"]).size().unstack(fill_value=0)
    for col in ["Positive", "Neutral", "Negative"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

    ax = pivot_pct[["Positive", "Neutral", "Negative"]].plot(
        kind="bar", figsize=(9, 5),
        color=["#4CAF50", "#FFC107", "#F44336"], edgecolor="black",
    )
    ax.set_title("Sentiment by Language (Top 5)")
    ax.set_xlabel("Language")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title="Sentiment")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "sentiment_by_language.png"), dpi=150)
    plt.close()

    print(f"Language & sentiment plots saved to {RESULTS_DIR}/")
    return sample


def extract_emojis(text: str) -> list:
    return [ch for ch in str(text) if ch in emoji.EMOJI_DATA]


def replace_emojis_with_text(text: str) -> str:
    result = []
    for ch in text:
        if ch in EMOJI_SENTIMENT:
            result.append(EMOJI_SENTIMENT[ch])
        elif ch in emoji.EMOJI_DATA:
            result.append(emoji.demojize(ch).replace(":", " ").replace("_", " "))
        else:
            result.append(ch)
    return "".join(result)


def score_with_and_without_emoji(text: str) -> dict:
    text = str(text)
    text_no_emoji = emoji.replace_emoji(text, replace="")
    score_no_emoji = VADER.polarity_scores(text_no_emoji)["compound"]
    text_with_emoji = replace_emojis_with_text(text)
    score_with_emoji = VADER.polarity_scores(text_with_emoji)["compound"]

    def to_label(s):
        if s >= 0.05: return "Positive"
        elif s <= -0.05: return "Negative"
        return "Neutral"

    return {
        "emojis_found": extract_emojis(text),
        "score_no_emoji": round(score_no_emoji, 4),
        "label_no_emoji": to_label(score_no_emoji),
        "score_with_emoji": round(score_with_emoji, 4),
        "label_with_emoji": to_label(score_with_emoji),
        "score_delta": round(score_with_emoji - score_no_emoji, 4),
    }


def run_emoji_analysis(df: pd.DataFrame, text_col: str = "text",
                        sample_n: int = 2000) -> pd.DataFrame:
    print("\nRunning Emoji-Aware Sentiment Analysis...")
    sample = df.sample(min(sample_n, len(df)), random_state=42).copy().reset_index(drop=True)

    sample["emoji_list"] = sample[text_col].apply(extract_emojis)
    emoji_df = sample[sample["emoji_list"].map(len) > 0].copy().reset_index(drop=True)
    print(f"  Found {len(emoji_df)} emoji-containing reviews (out of {len(sample)} sampled).")

    if len(emoji_df) == 0:
        print("  No emojis found in sample. Skipping emoji analysis.")
        return emoji_df

    score_no_list, label_no_list, score_with_list, label_with_list, delta_list = [], [], [], [], []
    for text in emoji_df[text_col]:
        result = score_with_and_without_emoji(text)
        score_no_list.append(result["score_no_emoji"])
        label_no_list.append(result["label_no_emoji"])
        score_with_list.append(result["score_with_emoji"])
        label_with_list.append(result["label_with_emoji"])
        delta_list.append(result["score_delta"])

    emoji_df["score_no_emoji"] = score_no_list
    emoji_df["label_no_emoji"] = label_no_list
    emoji_df["score_with_emoji"] = score_with_list
    emoji_df["label_with_emoji"] = label_with_list
    emoji_df["score_delta"] = delta_list

    all_emojis = [e for sublist in emoji_df["emoji_list"] for e in sublist]
    top_emojis = Counter(all_emojis).most_common(15)

    if top_emojis:
        emojis, counts = zip(*top_emojis)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(emojis)), counts, color="#FF9800", edgecolor="black")
        ax.set_xticks(range(len(emojis)))
        ax.set_xticklabels(emojis, fontsize=14)
        ax.set_title("Top 15 Most Frequent Emojis in Reviews")
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "top_emojis.png"), dpi=150)
        plt.close()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(emoji_df["score_delta"], bins=30, color="#9C27B0", edgecolor="black", alpha=0.8)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_title("Sentiment Score Change Due to Emojis")
    ax.set_xlabel("Score Delta (with emoji - without emoji)")
    ax.set_ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "emoji_score_delta.png"), dpi=150)
    plt.close()

    changed = (emoji_df["label_no_emoji"] != emoji_df["label_with_emoji"]).sum()
    total = len(emoji_df)
    print(f"  Emoji context changed sentiment label in {changed}/{total} reviews ({100*changed/total:.1f}%)")

    comparison = pd.DataFrame({
        "Without Emoji": emoji_df["label_no_emoji"].value_counts(),
        "With Emoji": emoji_df["label_with_emoji"].value_counts(),
    }).fillna(0)

    ax = comparison.plot(kind="bar", figsize=(7, 4), color=["#607D8B", "#FF9800"], edgecolor="black")
    ax.set_title("Sentiment Distribution: With vs Without Emoji Context")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "emoji_sentiment_comparison.png"), dpi=150)
    plt.close()

    print(f"Emoji analysis plots saved to {RESULTS_DIR}/")
    return emoji_df
multilingual_emoji.py
Displaying aspect_sentiment.py.