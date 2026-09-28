"""
Evaluate and compare two NER checkpoints on the hand-corrected noisy-tweet
eval sets (noisy_eval_hau.conll, noisy_eval_ibo.conll, noisy_eval_pcm.conll,
noisy_eval_yor.conll).

This is the actual evidence for the code-mixing adaptation: does the
continued-fine-tuned model (ner_model_v2_codemixed) score higher on real,
noisy, code-mixed tweets than the original baseline (ner_model_final)?

Run this in the same Colab session (or a fresh one with the same files
uploaded) after continue_training.py has finished.

Before running:
  1. Make sure your hand-corrected noisy_eval_*.conll files are in the
     current directory (upload them the same way you uploaded the others).
     If you haven't finished hand-correcting yet, you can run this now
     against the BIO-fixed-only versions to get a placeholder number, then
     re-run once corrections are done -- the corrected labels are the
     ground truth this script scores predictions against, so more accurate
     corrections = a more trustworthy F1 number.
  2. Make sure both model checkpoints are available (ner_model_final,
     and ner_model_v2_codemixed from Google Drive).

Usage:
    python evaluate_noisy.py
"""

import re
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
import evaluate

# ---------------------------------------------------------------------------
# Config -- adjust paths if yours differ
# ---------------------------------------------------------------------------
LANGS = ["hau", "ibo", "pcm", "yor"]

# Path template for the hand-corrected noisy eval files.
NOISY_EVAL_FILES = {lang: f"./noisy_eval_{lang}.conll" for lang in LANGS}

MODELS = {
    "baseline (ner_model_final)": "./ner_model_final",
    "code-mixing adapted (ner_model_v2_codemixed)": "/content/drive/MyDrive/ner_model_v2_codemixed",
}

MAX_LENGTH = 128
BATCH_SIZE = 32

LABEL_LIST = [
    "B-DATE", "B-LOC", "B-ORG", "B-PER",
    "I-DATE", "I-LOC", "I-ORG", "I-PER",
    "O",
]
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ---------------------------------------------------------------------------
# Read CoNLL files (same format as noisy_eval_*.conll: "token tag" per line,
# blank line between tweets)
# ---------------------------------------------------------------------------
def read_conll(path):
    with open(path, encoding="utf-8") as f:
        raw = f.read().replace("\r\n", "\n")
    sentences = []
    cur_tokens, cur_tags = [], []
    for line in raw.split("\n"):
        if not line.strip():
            if cur_tokens:
                sentences.append((cur_tokens, cur_tags))
                cur_tokens, cur_tags = [], []
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        tok, tag = parts
        cur_tokens.append(tok)
        cur_tags.append(tag)
    if cur_tokens:
        sentences.append((cur_tokens, cur_tags))
    return sentences


# ---------------------------------------------------------------------------
# Run a model over pre-tokenized sentences, return predicted tag sequences
# aligned back to the original word-level tokens (first-subword-wins, same
# convention used during training's tokenize_and_align_labels)
# ---------------------------------------------------------------------------
def predict_batch(tokenizer, model, batch_tokens):
    enc = tokenizer(
        batch_tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits
    preds = torch.argmax(logits, dim=-1).cpu().numpy()

    batch_pred_tags = []
    for i, tokens in enumerate(batch_tokens):
        word_ids = enc.word_ids(batch_index=i)
        pred_tags = []
        seen = set()
        for j, w_id in enumerate(word_ids):
            if w_id is None or w_id in seen:
                continue
            seen.add(w_id)
            pred_tags.append(ID2LABEL[preds[i][j]])
        # Guard against any tokenizer edge case producing a mismatched length
        if len(pred_tags) < len(tokens):
            pred_tags += ["O"] * (len(tokens) - len(pred_tags))
        elif len(pred_tags) > len(tokens):
            pred_tags = pred_tags[: len(tokens)]
        batch_pred_tags.append(pred_tags)
    return batch_pred_tags


def evaluate_model(model_path, data_by_lang):
    print(f"\nLoading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(DEVICE)
    model.eval()

    seqeval = evaluate.load("seqeval")
    per_lang_results = {}
    all_true, all_pred = [], []

    for lang, sentences in data_by_lang.items():
        lang_true, lang_pred = [], []
        for start in range(0, len(sentences), BATCH_SIZE):
            batch = sentences[start : start + BATCH_SIZE]
            batch_tokens = [tokens for tokens, _ in batch]
            batch_true_tags = [tags for _, tags in batch]
            batch_pred_tags = predict_batch(tokenizer, model, batch_tokens)
            lang_true.extend(batch_true_tags)
            lang_pred.extend(batch_pred_tags)

        lang_metrics = seqeval.compute(predictions=lang_pred, references=lang_true)
        per_lang_results[lang] = lang_metrics
        all_true.extend(lang_true)
        all_pred.extend(lang_pred)

    overall_metrics = seqeval.compute(predictions=all_pred, references=all_true)

    del model, tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return per_lang_results, overall_metrics


def fmt(m):
    return (
        f"P={m['overall_precision']:.4f}  "
        f"R={m['overall_recall']:.4f}  "
        f"F1={m['overall_f1']:.4f}  "
        f"Acc={m['overall_accuracy']:.4f}"
    )


if __name__ == "__main__":
    print("Loading hand-corrected noisy eval sets...")
    data_by_lang = {}
    for lang in LANGS:
        path = NOISY_EVAL_FILES[lang]
        sentences = read_conll(path)
        data_by_lang[lang] = sentences
        print(f"  {lang}: {len(sentences)} tweets loaded from {path}")

    all_results = {}
    for model_label, model_path in MODELS.items():
        per_lang, overall = evaluate_model(model_path, data_by_lang)
        all_results[model_label] = {"per_lang": per_lang, "overall": overall}

    print("\n" + "=" * 70)
    print("RESULTS: performance on hand-corrected noisy tweets")
    print("=" * 70)

    for lang in LANGS:
        print(f"\n--- {lang} ---")
        for model_label in MODELS:
            m = all_results[model_label]["per_lang"][lang]
            print(f"  {model_label}: {fmt(m)}")

    print("\n--- OVERALL (all 4 languages combined) ---")
    for model_label in MODELS:
        m = all_results[model_label]["overall"]
        print(f"  {model_label}: {fmt(m)}")

    labels = list(MODELS.keys())
    f1_before = all_results[labels[0]]["overall"]["overall_f1"]
    f1_after = all_results[labels[1]]["overall"]["overall_f1"]
    delta = f1_after - f1_before
    print(f"\nNoisy-tweet F1 change: {f1_before:.4f} -> {f1_after:.4f}  "
          f"({'+' if delta >= 0 else ''}{delta:.4f})")
    print("\nThis delta is the headline number for your thesis's code-mixing")
    print("adaptation contribution.")
