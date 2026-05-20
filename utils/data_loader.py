"""
utils/data_loader.py
Loads the Amazon Fine Food Reviews dataset and assigns sentiment labels.
Dataset: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
Expected CSV columns: Id, ProductId, UserId, ProfileName,
                      HelpfulnessNumerator, HelpfulnessDenominator,
                      Score, Time, Summary, Text
"""

import pandas as pd
import os


def load_dataset(filepath: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load the Amazon reviews CSV and derive sentiment labels from star ratings.

    Label mapping:
        Score 4-5  ->  Positive  (2)
        Score 3    ->  Neutral   (1)
        Score 1-2  ->  Negative  (0)

    Args:
        filepath    : path to Reviews.csv
        sample_size : optional int – take a random stratified sample
                      (useful for quick experiments)
    Returns:
        DataFrame with columns: text, summary, score, sentiment, sentiment_label
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'.\n"
            "Download from: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews\n"
            "Place Reviews.csv inside the data/ folder."
        )

    print(f"Loading dataset from {filepath} ...")
    df = pd.read_csv(filepath)

    # Keep only relevant columns
    df = df[["Score", "Summary", "Text"]].copy()
    df.columns = ["score", "summary", "text"]

    # Drop nulls
    df.dropna(subset=["text", "score"], inplace=True)
    df["score"] = df["score"].astype(int)

    # Derive sentiment label
    def score_to_sentiment(s):
        if s >= 4:
            return 2  # Positive
        elif s == 3:
            return 1  # Neutral
        else:
            return 0  # Negative

    df["sentiment"] = df["score"].apply(score_to_sentiment)
    df["sentiment_label"] = df["sentiment"].map({2: "Positive", 1: "Neutral", 0: "Negative"})

    if sample_size:
        df = (
            df.groupby("sentiment", group_keys=False)
              .apply(lambda x: x.sample(min(len(x), sample_size // 3), random_state=42))
              .reset_index(drop=True)
        )
        print(f"Sampled {len(df)} reviews (stratified).")

    print(f"Dataset loaded: {len(df)} reviews.")
    print(df["sentiment_label"].value_counts().to_string())
    return df


if __name__ == "__main__":
    df = load_dataset("data/Reviews.csv", sample_size=9000)
    print(df.head())