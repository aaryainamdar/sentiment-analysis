"""
models/traditional_ml.py
Traditional ML sentiment classifiers:
  - TF-IDF + Logistic Regression
  - TF-IDF + Multinomial Naive Bayes
  - TF-IDF + Linear SVM (bonus)

Outputs: classification reports, confusion matrices, saved model files.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)
from sklearn.pipeline import Pipeline

LABELS = ["Negative", "Neutral", "Positive"]
RESULTS_DIR = "results"
MODELS_DIR = "models/saved"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def build_pipelines() -> dict:
    """Return a dict of named sklearn Pipelines."""
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),   # unigrams + bigrams
        sublinear_tf=True,
        min_df=3,
    )

    return {
        "Logistic_Regression": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=3)),
            ("clf", LogisticRegression(max_iter=500, C=1.0,
                                        class_weight="balanced", random_state=42)),
        ]),
        "Naive_Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=3)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Linear_SVM": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=50_000, ngram_range=(1, 2),
                                      sublinear_tf=True, min_df=3)),
            ("clf", LinearSVC(max_iter=1000, C=1.0,
                               class_weight="balanced", random_state=42)),
        ]),
    }


def plot_confusion_matrix(cm: np.ndarray, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS, ax=ax)
    ax.set_title(f"Confusion Matrix – {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"cm_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {path}")


def train_and_evaluate(df: pd.DataFrame, text_col: str = "clean_text"):
    """
    Train all traditional ML pipelines and print/save results.

    Args:
        df       : DataFrame with 'clean_text' and 'sentiment' columns
        text_col : column to use as features
    """
    X = df[text_col].fillna("")
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}\n")

    pipelines = build_pipelines()
    summary = {}

    for name, pipeline in pipelines.items():
        print(f"{'='*50}")
        print(f"Training: {name}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=LABELS)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(report)

        plot_confusion_matrix(cm, name)

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(pipeline, model_path)
        print(f"  Model saved → {model_path}")

        summary[name] = acc

    # Summary bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(summary.keys(), summary.values(), color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_ylim(0, 1)
    ax.set_title("Traditional ML – Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    for bar, val in zip(bars, summary.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.3f}", ha="center", fontsize=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "traditional_ml_accuracy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nAccuracy comparison saved → {path}")

    return summary


def predict_single(text: str, model_name: str = "Logistic_Regression") -> str:
    """Load a saved model and predict sentiment for a single review."""
    from utils.preprocessor import clean_text
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pkl")
    pipeline = joblib.load(model_path)
    cleaned = clean_text(text)
    pred = pipeline.predict([cleaned])[0]
    return {2: "Positive", 1: "Neutral", 0: "Negative"}[pred]


if __name__ == "__main__":
    from utils.data_loader import load_dataset
    from utils.preprocessor import preprocess_dataframe

    df = load_dataset("data/Reviews.csv", sample_size=9000)
    df = preprocess_dataframe(df)
    train_and_evaluate(df)