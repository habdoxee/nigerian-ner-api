"""
Build a noisy-tweet evaluation set for the thesis's "noisy social-media NLP"
gap, using REAL tweets (not the clean MasakhaNER news text).

What this does:
  1. Pulls ~275 real tweets per language from NaijaSenti-Twitter (Hausa,
     Igbo, Nigerian-Pidgin, Yoruba) -- genuine noisy, code-mixed tweets.
  2. Runs your trained model on them to produce "silver" (automatic, likely
     imperfect) entity predictions.
  3. Writes one CoNLL-format .txt file per language, in the exact same
     token/tag-per-line format as your original training data, so you can:
       a) open each file in a text editor
       b) fix any wrong tags by hand (this is much faster than labeling
          from scratch, since most predictions will already be close)
       c) reload the corrected files with prepare_dataset.py's read_conll()
          function to build your noisy gold evaluation set

Usage:
    python build_noisy_eval_set.py

Requires:
    pip install datasets transformers torch

Output:
    ./noisy_eval/noisy_eval_hau.conll
    ./noisy_eval/noisy_eval_ibo.conll
    ./noisy_eval/noisy_eval_pcm.conll
    ./noisy_eval/noisy_eval_yor.conll

Each line is: <token> <predicted_tag>
Blank line separates tweets, matching your training data's CoNLL format.
"""

import os
import random
import requests
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_PATH = "./ner_model_final"   # your trained model (use the non-quantized
                                    # version here for the best possible silver
                                    # labels -- quantization is only needed for
                                    # the deployed API, not this one-off step)
OUTPUT_DIR = "./noisy_eval"
TWEETS_PER_LANGUAGE = 275
RANDOM_SEED = 42

# NaijaSenti-Twitter's HF config codes for each language
LANGUAGES = {
    "hau": "Hausa",
    "ibo": "Igbo",
    "pcm": "Nigerian Pidgin",
    "yor": "Yoruba",
}

LABEL_LIST = [
    "B-DATE", "B-LOC", "B-ORG", "B-PER",
    "I-DATE", "I-LOC", "I-ORG", "I-PER",
    "O",
]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

random.seed(RANDOM_SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load model + tokenizer once
# ---------------------------------------------------------------------------
print(f"Loading model from {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()


def predict_tags(tokens):
    """
    Given a list of whitespace-split tokens (words), returns a list of
    predicted BIO tags, one per token -- aligned back from subword pieces
    using the same first-subword convention as training.
    """
    encoding = tokenizer(
        tokens,
        truncation=True,
        max_length=128,
        is_split_into_words=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoding).logits
    predictions = torch.argmax(logits, dim=2)[0].tolist()

    word_ids = encoding.word_ids(batch_index=0)
    tags = []
    previous_word_idx = None
    for word_idx, pred_id in zip(word_ids, predictions):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            tags.append(ID2LABEL[pred_id])
        previous_word_idx = word_idx

    # Safety net: truncation could clip trailing words -- pad with "O" so
    # the tag list always matches the token list length for clean file output.
    while len(tags) < len(tokens):
        tags.append("O")
    return tags[: len(tokens)]


def clean_and_tokenize(text):
    """Simple whitespace tokenization, matching how the training data was built."""
    return text.strip().split()


# ---------------------------------------------------------------------------
# Process each language
# ---------------------------------------------------------------------------
def fetch_naijasenti_tweets(lang_code):
    """
    Downloads NaijaSenti's raw TSV files directly from GitHub, bypassing
    datasets.load_dataset() -- the HF Hub copy uses an old-style loading
    script that recent `datasets` versions (5.x+) no longer support.
    Combines train+dev+test since we only want raw tweet text.
    """
    base = f"https://raw.githubusercontent.com/hausanlp/NaijaSenti/main/data/annotated_tweets/{lang_code}"
    all_tweets = []
    for split_file in ["train.tsv", "dev.tsv", "test.tsv"]:
        url = f"{base}/{split_file}"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"  Warning: could not fetch {url} (status {resp.status_code})")
            continue
        lines = resp.text.splitlines()
        if not lines:
            continue
        header = lines[0].split("\t")
        try:
            tweet_idx = header.index("tweet")
        except ValueError:
            tweet_idx = 0
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) > tweet_idx and cols[tweet_idx].strip():
                all_tweets.append(cols[tweet_idx])
    return all_tweets


for lang_code, lang_name in LANGUAGES.items():
    print(f"\n--- {lang_name} ({lang_code}) ---")
    print("Downloading NaijaSenti tweets from GitHub...")
    all_texts = fetch_naijasenti_tweets(lang_code)
    if not all_texts:
        print(f"  No tweets fetched for {lang_code}, skipping.")
        continue
    print(f"  Fetched {len(all_texts)} total tweets (train+dev+test combined).")

    # Sample, dedupe, and cap at the target count
    all_texts = list(dict.fromkeys(all_texts))  # de-dupe, keep order
    random.shuffle(all_texts)
    sample = [t for t in all_texts if t and t.strip()][:TWEETS_PER_LANGUAGE]
    print(f"Sampled {len(sample)} tweets. Running model for silver labels...")

    output_path = os.path.join(OUTPUT_DIR, f"noisy_eval_{lang_code}.conll")
    with open(output_path, "w", encoding="utf-8") as f:
        for i, tweet_text in enumerate(sample):
            tokens = clean_and_tokenize(tweet_text)
            if not tokens:
                continue
            tags = predict_tags(tokens)
            for token, tag in zip(tokens, tags):
                f.write(f"{token} {tag}\n")
            f.write("\n")  # blank line separates tweets, matching CoNLL format
            if (i + 1) % 50 == 0:
                print(f"  {i + 1}/{len(sample)} tweets processed")

    print(f"Saved: {output_path}")

print("\nDone. Next steps:")
print("1. Open each noisy_eval_*.conll file in a text editor.")
print("2. Read through each tweet and correct any wrong tags by hand --")
print("   most predictions should already be close, so this is editing,")
print("   not labeling from scratch.")
print("3. Once corrected, load these files the same way prepare_dataset.py")
print("   loads your training data (its read_conll() function works as-is")
print("   on this format), then run your trained model against the")
print("   corrected labels to get a real noisy-tweet F1 score.")
