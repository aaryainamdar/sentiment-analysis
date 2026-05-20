# Sentiment Analysis — Amazon Fine Food Reviews

A modular, end-to-end NLP pipeline for sentiment analysis on the
[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) dataset.
Covers traditional ML, deep learning, aspect-based analysis, emoji handling, and multilingual support.

---

## Project structure

```
sentiment_analysis/
├── data/
│   └── Reviews.csv          ← the Kaggle CSV 
├── models/
│   ├── aspect_sentiment.py  ← ABSA with VADER + keyword matching
│   ├── deep_learning.py     ← Bidirectional LSTM (PyTorch)
│   ├── multilingual_emoji.py← Language detection + emoji-aware scoring
│   ├── traditional_ml.py    ← TF-IDF + Logistic Regression / NB / SVM
│   └── saved/               ← trained model checkpoints written here
├── utils/
│   ├── data_loader.py       ← load CSV, derive sentiment labels from star ratings
│   ├── eda.py               ← class distribution, word clouds, n-gram charts
│   └── preprocessor.py      ← lowercasing, stopword removal, lemmatisation
├── results/                 ← all output charts saved here as .png
├── main.py                  ← single entry point for the full pipeline
└── requirements.txt
```

---

## Setup

### 1 — Clone / unzip

```bash
unzip sentiment_analysis.zip
cd sentiment_analysis
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **PyTorch note**: the `requirements.txt` installs the CPU build of PyTorch.
> For GPU support visit https://pytorch.org/get-started/locally/ and install the
> appropriate CUDA wheel before running the pipeline.

### 4 — Download the dataset

1. Go to https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
2. Download `Reviews.csv`
3. Place it at `data/Reviews.csv`

---

## Running the pipeline

```bash
# Full run (9 000 sampled reviews, 10 BiLSTM epochs)
python main.py

# Custom sample size and epochs
python main.py --sample 5000 --epochs 5

# Skip the deep-learning step for a faster run
python main.py --skip-dl

# Use all reviews (slow — ~568 000 rows)
python main.py --sample 0
```

All charts are written to `results/` and trained models to `models/saved/`.

---

## What each module does

### `utils/data_loader.py`
Reads `Reviews.csv` and maps star ratings to three-class labels:

| Stars | Label    | Numeric |
|-------|----------|---------|
| 4–5   | Positive | 2       |
| 3     | Neutral  | 1       |
| 1–2   | Negative | 0       |

Supports stratified sampling via `sample_size`.

### `utils/preprocessor.py`
Text cleaning pipeline:
- Contraction expansion (`won't` → `will not`)
- Emoji handling — strip or convert to sentiment hint words
- HTML tag and URL removal
- Lowercasing, stopword removal, WordNet lemmatisation

### `utils/eda.py`
Generates:
- `class_distribution.png` — pie + bar chart of label counts
- `review_length_distribution.png` — word-count histogram by sentiment
- `wordcloud_positive/neutral/negative.png` — word clouds per class
- `top_1grams.png` / `top_2grams.png` — most frequent n-grams per class

### `models/traditional_ml.py`
Trains three TF-IDF pipelines (unigrams + bigrams, 50 k features):

| Model               | Notes                        |
|---------------------|------------------------------|
| Logistic Regression | balanced class weights       |
| Multinomial Naive Bayes | fast baseline            |
| Linear SVM          | often strongest on text      |

Saves confusion matrices and a model-accuracy bar chart to `results/`.
Pickled pipelines saved to `models/saved/*.pkl`.

### `models/deep_learning.py`
Bidirectional LSTM (PyTorch):
- Custom vocabulary (30 k tokens, min freq 2)
- 128-dim embeddings, 2-layer BiLSTM, 256 hidden units
- Dropout 0.4, inverse-frequency class weights, ReduceLROnPlateau scheduler
- Saves best checkpoint to `models/saved/bilstm_best.pt`
- Outputs loss/accuracy curves and confusion matrix

### `models/aspect_sentiment.py`
Aspect-Based Sentiment Analysis (ABSA):
Extracts sentences mentioning six aspects — **Quality, Price, Delivery,
Packaging, Taste, Service** — and scores each with VADER.
Outputs a stacked-bar chart and average-score horizontal bar chart.

### `models/multilingual_emoji.py`
Two independent analyses:

**Multilingual** — detects language with `langdetect`; uses VADER for English
and TextBlob for other languages. Plots language distribution and per-language
sentiment breakdown.

**Emoji-aware** — compares VADER scores with and without emoji context.
Reports how often emoji presence changes the predicted sentiment label.
Plots top-15 emojis and score-delta histogram.

---

## Output files

After a full run `results/` will contain:

```
class_distribution.png
review_length_distribution.png
wordcloud_positive.png
wordcloud_neutral.png
wordcloud_negative.png
top_1grams.png
top_2grams.png
cm_Logistic_Regression.png
cm_Naive_Bayes.png
cm_Linear_SVM.png
traditional_ml_accuracy.png
cm_BiLSTM.png
bilstm_training_curves.png
aspect_sentiment.png
aspect_avg_score.png
language_distribution.png
sentiment_by_language.png
top_emojis.png
emoji_score_delta.png
emoji_sentiment_comparison.png
```

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
vaderSentiment
torch
scikit-learn
textblob
emoji
nltk
joblib
tqdm
wordcloud
langdetect
```

---

## Notes

- NLTK resources (`punkt`, `stopwords`, `wordnet`) are downloaded automatically
  on first run.
- The `models/saved/` and `results/` directories are created automatically if
  they do not exist.
- To predict sentiment for a single review after training:

```python
from models.traditional_ml import predict_single
label = predict_single("This product is absolutely fantastic!", model_name="Logistic_Regression")
print(label)  # → Positive
```

---

## License

For educational and research use. Dataset terms: see Kaggle page linked above.
