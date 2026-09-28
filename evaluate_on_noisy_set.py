"""
Evaluate a trained model against your hand-corrected noisy tweet eval set,
producing real precision/recall/F1 on noisy, code-mixed Nigerian text.

Run this TWICE -- once per model -- to get the before/after comparison
that is your thesis's core evidence:

    python evaluate_on_noisy_set.py --model ./ner_model_final
    python evaluate_on_noisy_set.py --model ./ner_model_v2_codemixed

Requires:
    pip install seqeval torch transformers
"""

import argparse
import os
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
import evaluate

LABEL_LIST = [
    "B-DATE", "B-LOC", "B-ORG", "B-PER",
    "I-DATE", "I-LOC", "I-ORG", "I-PER",
    "O",
]
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

EVAL_SET_DIR = "./noisy_eval"


def read_conll(filepath):
    sentences = []
    tokens, tags = [], []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
            else:
                splits = line.split()
                if len(splits) >= 2:
                    tokens.append(splits[0])
                    tags.append(splits[-1])
        if tokens:
            sentences.append({"tokens": tokens, "ner_tags": tags})
    return sentences


def predict_tags(tokenizer, model, tokens, device):
    encoding = tokenizer(
        tokens, truncation=True, max_length=128,
        is_split_into_words=True, return_tensors="pt",
    ).to(device)
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
    while len(tags) < len(tokens):
        tags.append("O")
    return tags[: len(tokens)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model folder to evaluate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {args.model} (device: {device}) ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    model.to(device)
    model.eval()

    all_true, all_pred = [], []
    per_lang_results = {}

    for fname in sorted(os.listdir(EVAL_SET_DIR)):
        if not fname.endswith(".conll"):
            continue
        lang = fname.replace("noisy_eval_", "").replace(".conll", "")
        examples = read_conll(os.path.join(EVAL_SET_DIR, fname))
        print(f"  {fname}: {len(examples)} corrected examples")

        lang_true, lang_pred = [], []
        for ex in examples:
            pred_tags = predict_tags(tokenizer, model, ex["tokens"], device)
            lang_true.append(ex["ner_tags"])
            lang_pred.append(pred_tags)

        all_true.extend(lang_true)
        all_pred.extend(lang_pred)

        seqeval = evaluate.load("seqeval")
        lang_metrics = seqeval.compute(predictions=lang_pred, references=lang_true)
        per_lang_results[lang] = lang_metrics

    print("\n=== Per-language results ===")
    for lang, metrics in per_lang_results.items():
        print(f"{lang}: P={metrics['overall_precision']:.4f}  "
              f"R={metrics['overall_recall']:.4f}  "
              f"F1={metrics['overall_f1']:.4f}  "
              f"Acc={metrics['overall_accuracy']:.4f}")

    seqeval = evaluate.load("seqeval")
    overall = seqeval.compute(predictions=all_pred, references=all_true)
    print("\n=== OVERALL (all languages combined) ===")
    print(f"Precision: {overall['overall_precision']:.4f}")
    print(f"Recall:    {overall['overall_recall']:.4f}")
    print(f"F1:        {overall['overall_f1']:.4f}")
    print(f"Accuracy:  {overall['overall_accuracy']:.4f}")
    print(f"\nModel evaluated: {args.model}")


if __name__ == "__main__":
    main()
