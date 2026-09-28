""" 
Build a large-scale silver-labeled training set for code-mixing adaptation.

Pulls ~3000 real tweets per language from NaijaSenti-Twitter, and uses your
CURRENT trained model to auto-label them (silver labels -- no manual
correction at this scale, that's the point: self-training lets you adapt
to a new domain without hand-annotating thousands of new examples).

This is a different, LARGER, TRAINING-focused sibling of
build_noisy_eval_set.py, which pulled a small (275/language) sample for
manual correction as your trusted EVALUATION set. Keep both:
  - build_noisy_eval_set.py's output  -> small, hand-corrected, for eval
  - this script's output              -> large, silver-labeled, for training

Usage:
    python build_silver_training_set.py

Requires:
    pip install datasets transformers torch

Output:
    ./silver_training_data/silver_hau.conll
    ./silver_training_data/silver_ibo.conll
    ./silver_training_data/silver_pcm.conll
    ./silver_training_data/silver_yor.conll
"""

import os
import random
import io
import requests
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_PATH = "./ner_model_final"   # your current best model does the labeling
OUTPUT_DIR = "./silver_training_data"
TWEETS_PER_LANGUAGE = 3000
RANDOM_SEED = 7  # different seed than the eval set script, so samples don't overlap

# Exclude tweets already used in your manually-corrected eval set so training
# and evaluation data never overlap (would inflate your reported F1 falsely).
EVAL_SET_DIR = "./noisy_eval"

# Optional extra confidence layer: NaijaSenti's TSV files carry no geo/location
# field at all, so this can only check for EXPLICIT Nigeria-related keywords in
# the tweet text -- it cannot verify actual tweet origin. Defaults to OFF
# because most genuine Nigerian tweets don't explicitly name-drop the country,
# a state, or a city; turning this on WILL discard usable data, disproportionately
# from Pidgin (already your smallest pool). The four languages themselves are
# already a reasonably strong Nigeria signal per NaijaSenti's own curation
# methodology (see the paper: Muhammad et al. 2022). Set to True only if you
# want a stricter, smaller, more explicitly-Nigeria-signaled subset and are
# willing to accept fewer tweets, especially for Pidgin.
STRICT_NIGERIA_KEYWORD_FILTER = False

NIGERIA_KEYWORDS = [
    "nigeria", "naija", "9ja", "+234", "234",
    # Nigerian states + FCT (reuses your gazetteer's list)
    "abia", "adamawa", "akwa ibom", "anambra", "bauchi", "bayelsa", "benue",
    "borno", "cross river", "delta", "ebonyi", "edo", "ekiti", "enugu",
    "gombe", "imo", "jigawa", "kaduna", "kano", "katsina", "kebbi", "kogi",
    "kwara", "lagos", "nasarawa", "niger state", "ogun", "ondo", "osun", "oyo",
    "plateau", "rivers", "sokoto", "taraba", "yobe", "zamfara", "abuja", "fct",
    # major cities not already covered above
    "ibadan", "port harcourt", "benin city", "aba", "onitsha", "warri", "jos",
    "ilorin", "owerri", "calabar", "uyo", "maiduguri", "zaria", "abeokuta",
]


def mentions_nigeria(tweet_text):
    lowered = tweet_text.lower()
    return any(keyword in lowered for keyword in NIGERIA_KEYWORDS)


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

print(f"Loading model from {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()

# Use GPU if this happens to run on Colab; falls back to CPU locally.
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")


def load_already_used_tweets():
    """Read the eval set's original tweet text (reconstructed from tokens) so
    we can exclude those exact tweets from the training pool -- prevents
    train/eval leakage."""
    used = set()
    if not os.path.isdir(EVAL_SET_DIR):
        return used
    for fname in os.listdir(EVAL_SET_DIR):
        if not fname.endswith(".conll"):
            continue
        with open(os.path.join(EVAL_SET_DIR, fname), encoding="utf-8") as f:
            current_tokens = []
            for line in f:
                line = line.strip()
                if line == "":
                    if current_tokens:
                        used.add(" ".join(current_tokens))
                        current_tokens = []
                else:
                    current_tokens.append(line.split()[0])
            if current_tokens:
                used.add(" ".join(current_tokens))
    return used


def predict_tags_batch(token_lists):
    """Predict BIO tags for a batch of tokenized tweets at once (much faster
    than one-at-a-time, especially useful if running on a Colab GPU)."""
    encoding = tokenizer(
        token_lists,
        truncation=True,
        max_length=128,
        is_split_into_words=True,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**encoding).logits
    all_predictions = torch.argmax(logits, dim=2).tolist()

    results = []
    for i, tokens in enumerate(token_lists):
        word_ids = encoding.word_ids(batch_index=i)
        tags = []
        previous_word_idx = None
        for word_idx, pred_id in zip(word_ids, all_predictions[i]):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                tags.append(ID2LABEL[pred_id])
            previous_word_idx = word_idx
        while len(tags) < len(tokens):
            tags.append("O")
        results.append(tags[: len(tokens)])
    return results


already_used = load_already_used_tweets()
print(f"Excluding {len(already_used)} tweets already in the manual eval set.\n")

BATCH_SIZE = 32

def fetch_naijasenti_tweets(lang_code):
    """
    Downloads NaijaSenti's raw TSV files directly from GitHub, bypassing
    datasets.load_dataset() -- the HF Hub copy still uses an old-style
    loading script that recent `datasets` versions (5.x+) no longer support.
    Combines train+dev+test since we only want raw tweet text, not the
    sentiment-analysis split structure.
    """
    base = f"https://raw.githubusercontent.com/hausanlp/NaijaSenti/main/data/annotated_tweets/{lang_code}"
    all_tweets = []
    for split_file in ["train.tsv", "dev.tsv", "test.tsv"]:
        url = f"{base}/{split_file}"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"    Warning: could not fetch {url} (status {resp.status_code})")
            continue
        lines = resp.text.splitlines()
        if not lines:
            continue
        header = lines[0].split("\t")
        try:
            tweet_idx = header.index("tweet")
        except ValueError:
            tweet_idx = 0  # fall back to first column if header differs
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) > tweet_idx and cols[tweet_idx].strip():
                all_tweets.append(cols[tweet_idx])
    return all_tweets


for lang_code, lang_name in LANGUAGES.items():
    print(f"--- {lang_name} ({lang_code}) ---")
    all_texts = fetch_naijasenti_tweets(lang_code)
    if not all_texts:
        print(f"  No tweets fetched for {lang_code}, skipping.")
        continue
    print(f"  Fetched {len(all_texts)} total tweets (train+dev+test combined).")

    all_texts = list(dict.fromkeys(all_texts))  # de-dupe, preserve order
    random.shuffle(all_texts)

    if STRICT_NIGERIA_KEYWORD_FILTER:
        before_count = len(all_texts)
        all_texts = [t for t in all_texts if mentions_nigeria(t)]
        print(f"  Nigeria-keyword filter: {before_count} -> {len(all_texts)} tweets "
              f"({before_count - len(all_texts)} discarded, no explicit match)")

    sample = []
    for t in all_texts:
        if not t or not t.strip():
            continue
        if t.strip() in already_used:
            continue  # skip anything already in the eval set
        sample.append(t)
        if len(sample) >= TWEETS_PER_LANGUAGE:
            break

    print(f"Sampled {len(sample)} tweets (target was {TWEETS_PER_LANGUAGE}).")
    print("Running model in batches for silver labels...")

    output_path = os.path.join(OUTPUT_DIR, f"silver_{lang_code}.conll")
    with open(output_path, "w", encoding="utf-8") as f:
        for batch_start in range(0, len(sample), BATCH_SIZE):
            batch_texts = sample[batch_start: batch_start + BATCH_SIZE]
            batch_tokens = [t.strip().split() for t in batch_texts]
            batch_tokens = [tok for tok in batch_tokens if tok]  # drop empties
            if not batch_tokens:
                continue

            batch_tags = predict_tags_batch(batch_tokens)

            for tokens, tags in zip(batch_tokens, batch_tags):
                for token, tag in zip(tokens, tags):
                    f.write(f"{token} {tag}\n")
                f.write("\n")

            done = min(batch_start + BATCH_SIZE, len(sample))
            if done % 300 < BATCH_SIZE:
                print(f"  {done}/{len(sample)} processed")

    print(f"Saved: {output_path}\n")

print("Done. Next: run combine_datasets.py to merge this with your original")
print("clean training data before continuing fine-tuning.")
