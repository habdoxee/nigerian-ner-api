"""
Build a noisy-tweet evaluation set for the thesis's "noisy social-media NLP"
gap, using real tweets from NaijaSenti-Twitter (HausaNLP/NaijaSenti-Twitter),
pre-labeled with your trained model's predictions ("silver" labels) so you
only need to CORRECT them rather than annotate from scratch.

Pipeline:
    1. Load NaijaSenti-Twitter for hau, ibo, pcm, yor
    2. Sample ~250-300 tweets per language
    3. Tokenize each tweet the same way prepare_dataset.py's format expects
       (simple whitespace tokens, matching your existing CoNLL pipeline)
    4. Run your trained model over each tweet, one predicted tag per token
    5. Write out CoNLL-style files (token + predicted tag per line, blank
       line between tweets) -- same format your read_conll() function in
       prepare_dataset.py already knows how to parse, so once you've
       corrected the tags, this becomes a normal evaluation set with zero
       extra parsing code needed.
    6. Also write a human-readable reference file with the original raw
       tweet text next to each block, so you know what you're correcting.

Usage:
    python generate_silver_ner.py

Requires:
    pip install datasets transformers torch
"""

import os
import random
from datasets import load_dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
import torch

MODEL_PATH = "./ner_model_final"  # use the full-precision model here for best label quality
OUTPUT_DIR = "./noisy_eval_set"
SAMPLES_PER_LANGUAGE = 275  # middle of your 250-300 range
LANGUAGES = ["hau", "ibo", "pcm", "yor"]
MAX_LENGTH = 128

LABEL_LIST = [
    "B-DATE", "B-LOC", "B-ORG", "B-PER",
    "I-DATE", "I-LOC", "I-ORG", "I-PER",
    "O",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading model and tokenizer from {MODEL_PATH} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
model.eval()


def predict_tags_for_tokens(tokens):
    """
    Runs the model over a pre-tokenized (whitespace-split) sentence and
    returns one predicted label per input token -- mirroring exactly the
    word-alignment logic used during training (tokenize_and_align_labels
    in train.py), so the output format matches what the model was trained
    to be evaluated against.
    """
    encoding = tokenizer(
        tokens,
        truncation=True,
        max_length=MAX_LENGTH,
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
            tags.append(LABEL_LIST[pred_id])
        previous_word_idx = word_idx

    # Safety net in case of truncation cutting off trailing tokens
    while len(tags) < len(tokens):
        tags.append("O")

    return tags[: len(tokens)]


def simple_tokenize(text):
    """
    Whitespace tokenization, matching the granularity prepare_dataset.py's
    CoNLL files use. Deliberately not doing aggressive cleaning here --
    the noise (missing punctuation spacing, elongated words, etc.) is
    exactly what this evaluation set is supposed to capture.
    """
    return text.split()


for lang in LANGUAGES:
    print(f"\nLoading NaijaSenti-Twitter [{lang}] ...")
    ds = load_dataset("HausaNLP/NaijaSenti-Twitter", lang, split="train")

    # Filter out empty/very short tweets, then sample
    candidates = [ex["tweet"] for ex in ds if ex.get("tweet") and len(ex["tweet"].split()) >= 4]
    random.seed(42)
    sampled = random.sample(candidates, min(SAMPLES_PER_LANGUAGE, len(candidates)))
    print(f"  Sampled {len(sampled)} tweets (requested {SAMPLES_PER_LANGUAGE})")

    conll_path = os.path.join(OUTPUT_DIR, f"{lang}_silver.conll")
    reference_path = os.path.join(OUTPUT_DIR, f"{lang}_reference.txt")

    with open(conll_path, "w", encoding="utf-8") as conll_f, \
         open(reference_path, "w", encoding="utf-8") as ref_f:

        for i, tweet_text in enumerate(sampled):
            tokens = simple_tokenize(tweet_text)
            if not tokens:
                continue
            tags = predict_tags_for_tokens(tokens)

            # Reference file: numbered to match the Nth block in the CoNLL
            # file below (1st tweet = 1st block, in order), so you can look
            # up "what was this tweet actually saying" while correcting.
            ref_f.write(f"[{lang}-{i}] {tweet_text}\n")

            # CoNLL file: token + tag per line, blank line between tweets.
            # No comment/ID lines here on purpose -- this keeps the file in
            # the exact plain format your existing read_conll() function
            # already parses, so no changes needed there later.
            for tok, tag in zip(tokens, tags):
                conll_f.write(f"{tok}\t{tag}\n")
            conll_f.write("\n")

    print(f"  Wrote {conll_path}")
    print(f"  Wrote {reference_path}")

print(f"\nDone. Silver-labeled files are in {OUTPUT_DIR}/")
print("Next step: open each *_silver.conll file and correct the predicted")
print("tags by hand, using the matching *_reference.txt file to see the")
print("original tweet text for context.")
