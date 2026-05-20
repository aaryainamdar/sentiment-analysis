"""
utils/preprocessor.py
NLP text cleaning pipeline for sentiment analysis.
Handles standard text, emojis, and multilingual content.
"""

import re
import nltk
import emoji
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources (safe to call multiple times)
for resource in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Emoji-to-sentiment hint mapping (used in emoji-aware analysis)
EMOJI_SENTIMENT = {
    "😊": "happy", "😀": "happy", "😍": "love", "❤️": "love",
    "👍": "good", "✅": "good", "🎉": "great", "🌟": "great",
    "😢": "sad", "😭": "cry", "😡": "angry", "👎": "bad",
    "💔": "heartbreak", "😞": "disappointed", "🤮": "disgusting",
    "😲": "surprised", "🤔": "thinking", "😐": "neutral",
}


def extract_emojis(text: str) -> list:
    """Return a list of emoji characters found in text."""
    return [ch for ch in text if ch in emoji.EMOJI_DATA]


def replace_emojis_with_text(text: str) -> str:
    """Replace each emoji with its sentiment hint word (if mapped) or demojize it."""
    result = []
    for ch in text:
        if ch in EMOJI_SENTIMENT:
            result.append(EMOJI_SENTIMENT[ch])
        elif ch in emoji.EMOJI_DATA:
            # Fallback: use emoji name (e.g. :smiling_face:)
            result.append(emoji.demojize(ch).replace(":", " ").replace("_", " "))
        else:
            result.append(ch)
    return "".join(result)


def clean_text(text: str, keep_emojis: bool = False) -> str:
    """
    Full preprocessing pipeline:
      1. Lowercase
      2. Expand contractions (basic)
      3. Handle emojis
      4. Remove URLs, HTML tags, special characters
      5. Tokenize
      6. Remove stopwords
      7. Lemmatize

    Args:
        text        : raw review string
        keep_emojis : if True, emojis are converted to text hints;
                      if False, emojis are stripped
    Returns:
        Cleaned string
    """
    if not isinstance(text, str):
        return ""

    # Basic contraction expansion
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'ve": " have", "'m": " am",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)

    # Handle emojis
    if keep_emojis:
        text = replace_emojis_with_text(text)
    else:
        text = emoji.replace_emoji(text, replace="")

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOP_WORDS and len(tok) > 2
    ]

    return " ".join(tokens)


def preprocess_dataframe(df, text_col: str = "text", keep_emojis: bool = False):
    """
    Apply clean_text to an entire DataFrame column.

    Args:
        df          : pandas DataFrame
        text_col    : column containing raw review text
        keep_emojis : passed to clean_text
    Returns:
        DataFrame with new 'clean_text' column
    """
    print(f"Preprocessing {len(df)} reviews (keep_emojis={keep_emojis}) ...")
    from tqdm import tqdm
    tqdm.pandas()
    df = df.copy()
    df["clean_text"] = df[text_col].progress_apply(
        lambda t: clean_text(t, keep_emojis=keep_emojis)
    )
    print("Preprocessing complete.")
    return df


if __name__ == "__main__":
    samples = [
        "This product is absolutely AMAZING!! 😍❤️ Best buy ever!!!",
        "Terrible quality. Won't buy again 😡👎",
        "It's okay, nothing special 😐",
    ]
    for s in samples:
        print(f"Original : {s}")
        print(f"Cleaned  : {clean_text(s, keep_emojis=True)}")
        print(f"Emojis   : {extract_emojis(s)}")
        print()
