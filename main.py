"""
main.py
End-to-end sentiment analysis pipeline for Amazon Fine Food Reviews.

Usage:
    python main.py [--data data/Reviews.csv] [--sample 9000] [--epochs 10]

Steps executed:
    1. Load & label dataset
    2. Preprocess text
    3. Exploratory Data Analysis (EDA)
    4. Traditional ML models  (LR, Naive Bayes, SVM)
    5. Deep Learning model    (BiLSTM)
    6. Aspect-Based Sentiment Analysis (ABSA)
    7. Multilingual + Emoji analysis
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis Pipeline")
    parser.add_argument("--data",    default="data/Reviews.csv", help="Path to Reviews.csv")
    parser.add_argument("--sample",  type=int, default=9000,     help="Number of reviews to sample (0 = all)")
    parser.add_argument("--epochs",  type=int, default=10,        help="BiLSTM training epochs")
    parser.add_argument("--skip-dl", action="store_true",         help="Skip deep learning (faster run)")
    args = parser.parse_args()

    sample_size = args.sample if args.sample > 0 else None

    # ── 1. Load ────────────────────────────────────────────────────────────────
    from utils.data_loader import load_dataset
    df = load_dataset(args.data, sample_size=sample_size)

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    from utils.preprocessor import preprocess_dataframe
    df = preprocess_dataframe(df, text_col="text")

    # ── 3. EDA ─────────────────────────────────────────────────────────────────
    from utils.eda import run_eda
    run_eda(df, text_col="text", clean_col="clean_text")

    # ── 4. Traditional ML ──────────────────────────────────────────────────────
    from models.traditional_ml import train_and_evaluate
    ml_summary = train_and_evaluate(df, text_col="clean_text")
    print("\nTraditional ML accuracy summary:")
    for name, acc in ml_summary.items():
        print(f"  {name:25s} {acc:.4f}")

    # ── 5. Deep Learning ───────────────────────────────────────────────────────
    if not args.skip_dl:
        from models.deep_learning import train_lstm
        train_lstm(df, text_col="clean_text", epochs=args.epochs)
    else:
        print("\n[Skipping deep learning — --skip-dl flag set]")

    # ── 6. Aspect-Based Sentiment Analysis ─────────────────────────────────────
    from models.aspect_sentiment import run_aspect_analysis
    run_aspect_analysis(df, text_col="text")

    # ── 7. Multilingual + Emoji ────────────────────────────────────────────────
    from models.multilingual_emoji import run_multilingual_analysis, run_emoji_analysis
    run_multilingual_analysis(df, text_col="text")
    run_emoji_analysis(df, text_col="text")

    print("\n✓ Pipeline complete. All outputs saved to results/ and models/saved/")


if __name__ == "__main__":
    main()
